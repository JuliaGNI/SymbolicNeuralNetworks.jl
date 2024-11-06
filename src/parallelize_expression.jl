@kernel function parallelize_expression_inplace_kernel!(output::AT, input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
    j, k = @index(Global, NTuple)

    expression!(output[:, j, k], input[:, j, k], ps)
    nothing
end

function parallelize_expression_inplace(expression!::Base.Callable)

    function parallelized_expression_inplace(output::AT, input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
        backend = KernelAbstractions.get_backend(output)
        kernel! = parallelize_expression_inplace_kernel!(backend)
        kernel!(output, input, ps, expression!, ndrange=(size(output, 2), size(output, 3)))
        nothing
    end

    function parallelized_expression_inplace(output::VT, input::VT, ps) where {T, VT <: AbstractVector{T}}
        expression!(output, input, ps)
    end

    parallelized_expression_inplace
end

@kernel function parallelize_expression_kernel!(output::AT, input::AT, ps, expression::Base.Callable) where {T, AT <: AbstractArray{T, 3}}
    j, k = @index(Global, NTuple)

    output[:, j, k] .= expression(input[:, j, k], ps)
    nothing
end

@kernel function parallelize_expression_differential_kernel!(dinput::AT, dnt::AbstractMatrix{<:NamedTuple}, doutput::AT, input::AT, ps, dpullback::Base.Callable) where {T, AT <: AbstractArray{T, 3}}
    j, k = @index(Global, NTuple)

    din, dnt_jk = dpullback(input[:, j, k], ps)(doutput[:, j, k])
    dinput[:, j, k] .= din
    for key in keys(dnt[j, k])
        dnt[j, k][key] .= dnt_jk[key]
    end

    nothing
end

@kernel function parallelize_expression_differential_kernel!(dinput::AT, dt::AbstractMatrix{<:Tuple}, doutput::AT, input::AT, ps, dpullback::Base.Callable) where {T, AT <: AbstractArray{T, 3}}
    j, k = @index(Global, NTuple)

    din, dt_jk = dpullback(input[:, j, k], ps)(doutput[:, j, k])
    dinput[:, j, k] .= din
    for i in 1:length(dt[j, k])
        for key in keys(dt[j, k][i])
            dt[j, k][i][key] .= dt_jk[i][key]
        end
    end

    nothing
end

_sum(A::AbstractMatrix; kwargs...) = sum(A; kwargs...)
function _sum(A::AbstractMatrix{<:NamedTuple}; kwargs...)
    keys_of_nt = keys(A[1, 1])
    entries = ()
    for key in keys_of_nt
        matrix_for_key = [A[j, k][key] for j ∈ axes(A, 1), k ∈ axes(A, 2)]
        entries = (entries..., _sum(matrix_for_key; kwargs...))
    end
    NamedTuple{keys_of_nt}(entries)
end

function _sum(A::AbstractMatrix{<:Tuple}; kwargs...)
    indices = 1:length(A[1, 1])
    entries = ()
    for index in indices
        matrix_for_index = [A[j, k][index] for j ∈ axes(A, 1), k ∈ axes(A, 2)]
        entries = (entries..., _sum(matrix_for_index; kwargs...))
    end
    entries
end

function parallelize_expression(expression::Base.Callable)

    pb(z, ps) =  Zygote.pullback(expression, z, ps)[2]

    function parallelized_expression(input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
        backend = KernelAbstractions.get_backend(input)
        output = KernelAbstractions.zeros(backend, T, size(input)...)
        kernel! = parallelize_expression_kernel!(backend)
        kernel!(output, input, ps, expression; ndrange = (size(input, 2), size(input, 3)))
        output
    end

    function parallelized_expression(input::VT, ps) where {T, VT <: AbstractVector{T}}
        expression(input, ps)
    end

    parallelized_expression, pb
end

function parallelize_pullback!(parallelized_expression, pb)
    @eval function ChainRulesCore.rrule(::typeof(parallelized_expression), input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
        output = parallelized_expression(input, ps)
        function parallelized_expression_pullback(doutput::AT)
            f̄ = NoTangent()
            backend = KernelAbstractions.get_backend(doutput)
            dinput = zero(input)
            dnt = [deepcopy(ps) for _ ∈ axes(input, 2), _ ∈ axes(input, 3)]
            kernel! = parallelize_expression_differential_kernel!(backend)
            kernel!(dinput, dnt, doutput, input, ps, pb; ndrange = (size(input, 2), size(input, 3)))
            dnt_final = _sum(dnt)
            f̄, dinput, dnt_final
        end
        output, parallelized_expression_pullback
    end
    nothing
end