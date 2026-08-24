@doc raw"""
    AbstractBatchedFunction{NDATA, R}

An executable function built from a symbolic equation, as returned by [`build_nn_function`](@ref).

It wraps a kernel that evaluates a *single* sample (see [`build_kernel`](@ref) and
[`build_kernel!`](@ref)) and adds everything the kernel does not do: iterating over a batch,
allocating and shaping the result, and accepting a single sample or a three-dimensional batch
instead of a matrix.

`NDATA` is the number of data arguments, so an instance is called as

```julia
f(input, ps)            # NDATA = 1
f(input, output, ps)    # NDATA = 2
f(x1, …, xNDATA, ps)    # in general
```

There is no bound on `NDATA`. One data argument is the network input, a second is typically the target
output of a loss, and the layerwise pullback uses one per entry of a layer's seam plus one for the
output sensitivities — see [`seam_interface`](@ref).

and `R` is how the per-sample results are combined — `hcat` or `+`.

# Result shapes

For an equation of size ``(m, n, \ldots)`` and a batch of ``N`` samples:

| data arguments | `R`    | result |
|----------------|--------|--------|
| vectors        | either | the shape of the equation |
| ``d\times{}N`` matrices | `hcat` | ``m\times(n\cdot\ldots\cdot{}N)`` |
| ``d\times{}N`` matrices | `+`    | the shape of the equation |
| ``d\times{}N_1\times{}N_2`` arrays | `hcat` | ``m\times{}N_1\times{}N_2`` (vector- or scalar-valued equations only) |
| ``d\times{}N_1\times{}N_2`` arrays | `+`    | the shape of the equation |

A scalar-valued equation counts as ``m = 1`` here, so batching it with `hcat` gives a ``1\times{}N``
matrix.

All data arguments must have the same number of dimensions and the same batch size.
"""
abstract type AbstractBatchedFunction{NDATA, R} <: Function end

"""
    OutOfPlaceBatchedFunction{NDATA}(kernel, equation_size, reduction)

An [`AbstractBatchedFunction`](@ref) that evaluates its kernel once per batch column and combines
the results with `Base.reduce`. It allocates an array per column, but — unlike
[`InPlaceBatchedFunction`](@ref) — it does not mutate anything and can therefore be differentiated
by `Zygote`.

This is what [`build_nn_function`](@ref) returns for `inplace = false`, and for scalar-valued
equations, for which `Symbolics.build_function` emits no in-place form.
"""
struct OutOfPlaceBatchedFunction{NDATA, R, KT, N} <: AbstractBatchedFunction{NDATA, R}
    kernel::KT
    equation_size::NTuple{N, Int}
    reduction::R
end

"""
    InPlaceBatchedFunction{NDATA}(kernel!, equation_size, reduction)

An [`AbstractBatchedFunction`](@ref) that allocates the result once and lets its kernel write every
batch column into it. This is the default of [`build_nn_function`](@ref); it costs a single
allocation per call rather than one per column, at the price of not being differentiable by
`Zygote` (`Mutating arrays is not supported`). Forward-mode AD works either way, as the element type
of the preallocated array is promoted over the inputs — see [`promoted_eltype`](@ref).
"""
struct InPlaceBatchedFunction{NDATA, R, KT, N} <: AbstractBatchedFunction{NDATA, R}
    kernel!::KT
    equation_size::NTuple{N, Int}
    reduction::R
end

for T in (:OutOfPlaceBatchedFunction, :InPlaceBatchedFunction)
    @eval function $T{NDATA}(kernel::KT, equation_size::NTuple{N, Int}, reduction::R) where {NDATA, KT, N, R}
        $T{NDATA, R, KT, N}(kernel, equation_size, reduction)
    end
end

# The two common arities are written out, so that `(input, ps)` and `(input, output, ps)` are calls
# the compiler sees the arity of. Beyond them the arguments are collected, which is what the layerwise
# pullback's kernels need — a layer that carries data alongside the state takes one argument per entry
# of its seam plus the output sensitivities; see `seam_interface`.
(f::AbstractBatchedFunction{1})(input, ps) = evaluate_batch(f, (input,), ps)
(f::AbstractBatchedFunction{2})(input, output, ps) = evaluate_batch(f, (input, output), ps)

function (f::AbstractBatchedFunction{NDATA})(args...) where {NDATA}
    length(args) == NDATA + 1 || throw(ArgumentError(
        "this function takes $(NDATA) data argument(s) and the parameters, i.e. $(NDATA + 1) " *
        "arguments in total, got $(length(args))."))
    evaluate_batch(f, args[1:NDATA], last(args))
end

function Base.show(io::IO, f::AbstractBatchedFunction{NDATA}) where {NDATA}
    arguments = NDATA == 1 ? "(input, ps)" :
                NDATA == 2 ? "(input, output, ps)" : "(x1, …, x$(NDATA), ps)"
    print(io, nameof(typeof(f)), " ", arguments, " for an equation of size ", f.equation_size,
          ", reduced with ", f.reduction)
end

"""
    evaluate_batch(f, data, ps)

Evaluate an [`AbstractBatchedFunction`](@ref) over the batch held in the tuple of data arguments
`data`. There is one method per rank of the data arguments; the matrix one is the workhorse and the
other two reduce to it.
"""
function evaluate_batch end

# --- a single sample ---------------------------------------------------------------------------
# The result keeps the shape of the equation rather than being given a batch dimension, as the
# equation may itself be matrix-valued.

function evaluate_batch(f::AbstractBatchedFunction, data::NTuple{N, AbstractVector}, ps) where {N}
    evaluate_sample(f, map(_as_column, data), ps)
end

function evaluate_sample(f::OutOfPlaceBatchedFunction, data::Tuple, ps)
    f.kernel(data..., ps, 1)
end

function evaluate_sample(f::InPlaceBatchedFunction, data::Tuple, ps)
    out = allocate_single_output(promoted_eltype(data..., ps), f.equation_size, f.reduction)
    f.kernel!(out, data..., ps, 1)
    out
end

_as_column(x::AbstractVector) = reshape(x, length(x), 1)

# --- a batch of samples ------------------------------------------------------------------------

function evaluate_batch(f::OutOfPlaceBatchedFunction, data::NTuple{N, AbstractMatrix}, ps) where {N}
    columns = _batch_axis(data)
    # `Base.reduce` has nothing to fold for an empty batch; the allocators produce the same empty
    # result the in-place path returns for one.
    isempty(columns) && return allocate_batch_output(promoted_eltype(data..., ps), f.equation_size,
                                                     0, f.reduction)
    # `Base.reduce(hcat, ::Vector)` sizes the result once (linear in the batch size), whereas
    # `mapreduce(…, hcat, …)` folds left to right and recopies the growing accumulator once per
    # column (quadratic in it).
    Base.reduce(f.reduction, [f.kernel(data..., ps, k) for k in columns])
end

function evaluate_batch(f::InPlaceBatchedFunction, data::NTuple{N, AbstractMatrix}, ps) where {N}
    columns = _batch_axis(data)
    out = allocate_batch_output(promoted_eltype(data..., ps), f.equation_size, length(columns), f.reduction)
    for k in columns
        f.kernel!(out, data..., ps, k)
    end
    out
end

function _batch_axis(data::Tuple)
    columns = axes(first(data), 2)
    all(x -> axes(x, 2) == columns, data) || throw(DimensionMismatch(
        "the data arguments have different batch sizes: $(map(x -> size(x, 2), data))."))
    columns
end

# --- a batch with two batch dimensions ----------------------------------------------------------
# The two trailing dimensions are flattened into one, the result of which is unflattened again when
# the samples were concatenated (with `+` they were summed, so there is no batch dimension left).

function evaluate_batch(f::AbstractBatchedFunction, data::NTuple{N, AbstractArray{<:Number, 3}}, ps) where {N}
    batch_size = (size(first(data), 2), size(first(data), 3))
    flattened = evaluate_batch(f, map(_flatten_batch_dimensions, data), ps)
    _restore_batch_dimensions(flattened, f.equation_size, f.reduction, batch_size)
end

_flatten_batch_dimensions(x::AbstractArray{<:Number, 3}) = reshape(x, size(x, 1), size(x, 2) * size(x, 3))

# --- anything else ------------------------------------------------------------------------------
# Less specific than the three methods above, so it is only reached when the data arguments do not
# all have one of the supported ranks. Without it the caller gets a `MethodError` naming
# `evaluate_batch` and the full `RuntimeGeneratedFunction` type, which says nothing about the cause.

function evaluate_batch(::AbstractBatchedFunction, data::Tuple, ::Any)
    throw(ArgumentError(
        "the data arguments have to be all vectors (a single sample), all matrices (a batch), or " *
        "all three-dimensional arrays (a batch with two batch dimensions); got arguments of " *
        "$(length(data) == 1 ? "rank" : "ranks") $(join(map(ndims, data), ", "))."))
end

_restore_batch_dimensions(out, ::Tuple, ::typeof(+), ::Tuple) = out

function _restore_batch_dimensions(out, equation_size::Tuple, ::typeof(hcat), batch_size::Tuple)
    trailing_dimensions(equation_size) == 1 || throw(ArgumentError(two_batch_dimension_message(equation_size)))
    reshape(out, size(out, 1), batch_size...)
end

"""
    trailing_dimensions(equation_size)

The number of columns one sample of an equation of size `equation_size` occupies, i.e. the product of
all but its first dimension. A scalar-valued equation counts as one of size ``m = 1``, so its empty
size gives `1` — `Base.tail` would throw on it.
"""
trailing_dimensions(::Tuple{}) = 1
trailing_dimensions(equation_size::Tuple) = prod(Base.tail(equation_size); init = 1)

"""
    two_batch_dimension_message(equation_size)

The error text for an equation whose result already uses the second dimension. Shared with
[`unflatten_batch`](@ref), so that an entry of an equation set is rejected in the same words as the
same equation built on its own.
"""
two_batch_dimension_message(equation_size::Tuple) =
    "an equation of size $(equation_size) cannot be evaluated on a batch with two batch " *
    "dimensions: concatenating the results already uses the second dimension. Reshape the " *
    "input into a matrix, or use `reduce = +`."

# --- allocating the result ----------------------------------------------------------------------

"""
    promoted_eltype(args...)

The element type the generated code will produce, promoted over all inputs.

This is needed because the in-place kernels write into an array that has to be allocated *before*
they are called, so the element type cannot be taken from a result. Promoting over the inputs keeps
`Float32` parameters, symbolic (`Num`) inputs and `ForwardDiff.Dual` numbers working.

The walk over a nested parameter set is `NeuralNetworkParameters.parameter_eltype`, which promotes
over the leaves and reaches the storage of a structured one — so a parameter that keeps fewer numbers
than its interface shows contributes the element type of the numbers it actually keeps.

Note that this derives the element type from the *inputs*, not from the expression, which the
out-of-place path does instead. The two can differ: an equation over integer inputs and integer
parameters evaluates to a `Float64`, which no `Array{Int}` can hold, so the allocators widen an
integer type with `float`. A `Float32` network whose generated code contains a `Float64` constant is
rounded to `Float32` rather than widened — which is the behaviour one wants for the network, but is
worth being aware of.
"""
promoted_eltype(args...) = promote_type(map(parameter_eltype, args)...)

@doc raw"""
    allocate_batch_output(T, equation_size, batch_size, reduction)

Allocate the result of evaluating an equation of size `equation_size` over `batch_size` samples,
with the shape documented for [`AbstractBatchedFunction`](@ref).

`T` is widened with `float` when it is an integer type: it comes from [`promoted_eltype`](@ref),
i.e. from the inputs, and an equation over integer inputs generally does not evaluate to an integer.

A scalar-valued equation — `equation_size == ()` — counts as one of size ``m = 1``, so `hcat` gives a
``1\times{}N`` matrix and `+` a number. Those two methods are only reached for an *empty* batch: the
in-place path, which is what allocates a result up front, does not exist for a scalar equation.
"""
allocate_batch_output(::Type{T}, ::Tuple{}, ::Integer, ::typeof(+)) where {T} =
    zero(_float_if_integer(T))
allocate_batch_output(::Type{T}, equation_size::Tuple, ::Integer, ::typeof(+)) where {T} =
    zeros(_float_if_integer(T), equation_size...)
allocate_batch_output(::Type{T}, ::Tuple{}, batch_size::Integer, ::typeof(hcat)) where {T} =
    Array{_float_if_integer(T)}(undef, 1, batch_size)
allocate_batch_output(::Type{T}, equation_size::Tuple, batch_size::Integer, ::typeof(hcat)) where {T} =
    Array{_float_if_integer(T)}(undef, equation_size[1], trailing_dimensions(equation_size) * batch_size)

"""
    allocate_single_output(T, equation_size, reduction)

Allocate the result of evaluating an equation of size `equation_size` for a single sample. Unlike
[`allocate_batch_output`](@ref) this keeps the shape of the equation, as the result may itself be a
matrix.
"""
allocate_single_output(::Type{T}, equation_size::Tuple, ::typeof(+)) where {T} =
    zeros(_float_if_integer(T), equation_size...)
allocate_single_output(::Type{T}, equation_size::Tuple, ::typeof(hcat)) where {T} =
    Array{_float_if_integer(T)}(undef, equation_size...)

"""
    _float_if_integer(T)

`float(T)` for integer types, `T` for everything else. See [`allocate_batch_output`](@ref).
`Bool` is included, as `float(Bool)` is `Float64`.
"""
_float_if_integer(::Type{T}) where {T <: Integer} = float(T)
_float_if_integer(::Type{T}) where {T} = T
