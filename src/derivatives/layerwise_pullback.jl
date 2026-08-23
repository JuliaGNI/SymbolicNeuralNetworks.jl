# Building the pullback layer by layer instead of differentiating one inlined expression.
#
# `SymbolicPullback`'s original construction builds a single scalar expression for the loss of the
# whole network and differentiates *that* once per scalar parameter. Both halves multiply: the forward
# pass is inlined through `AbstractNeuralNetworks.applychain`, so layer k's expression contains layer
# k−1's once per element it reads and the loss expression is O(widthᵈᵉᵖᵗʰ); `symbolic_derivative` then
# walks the whole of it once per parameter. A four-layer width-16 network reaches a gradient
# expression of 2·10⁸ nodes and does not build.
#
# The composition is what is expensive, and a composition does not have to be inlined to be
# differentiated. Keeping a symbolic *seam* between the layers — fresh variables for each layer's
# input, rather than the expression of everything upstream — makes the symbolic material a sum over
# layers instead of a product:
#
#   | four layers | monolithic  | layerwise |
#   |-------------|------------:|----------:|
#   | width 4     |     388 700 |     2 520 |
#   | width 8     |   8 253 148 |    11 736 |
#   | width 16    | 209 455 964 |    68 760 |
#
# The composition then happens at *evaluation* time, where it costs a function call per layer.

@doc raw"""
    PassThroughLayer{N}()

A layer that returns its input unchanged, with no parameters. Used to obtain the loss as a function
of the network's *prediction* rather than of its input; see [`loss_expression`](@ref).

`AbstractNeuralNetworks` has no such interface: a `NetworkLoss` is applied as
`loss(model, ps, input, output)`, so the only way to ask it "what are you, as a function of the
prediction and the target?" is to hand it a model whose prediction *is* its input.
"""
struct PassThroughLayer{N} <: AbstractExplicitLayer{N, N} end

(::PassThroughLayer)(x, ps) = x

@doc raw"""
    loss_expression(loss, ŷ, y)

The symbolic expression of `loss` as a function of the prediction `ŷ` and the target `y`, from which
the layerwise pullback takes its seed ``\partial{}L/\partial\hat{y}``.

Returns `nothing` by default, which means "not declared" — the package then guesses the expression
with [`passthrough_expression`](@ref) and *checks the guess* before using it.

# Extending

Declare the expression for a loss whose relation between prediction and target the guess cannot
represent — one that carries extra data of its own, or that compares the prediction to the network's
input rather than to `output`, as an autoencoder loss does:

```julia
SymbolicNeuralNetworks.loss_expression(loss::MyLoss, ŷ, y) = ...
```

A declared expression is used as given. It is deliberately not checked against `loss` the way the
guess is: the reason to declare one is that the four-argument form means something the check would
assume it does not, so checking it against that assumption would reject exactly the methods this
exists for.
"""
# `ŷ` and `y` are deliberately left untyped, so that an overriding method written the way the
# docstring shows — `loss_expression(loss::MyLoss, ŷ, y)` — is strictly more specific than this one
# rather than ambiguous with it.
loss_expression(::NetworkLoss, ŷ, y) = nothing

"""
    passthrough_expression(loss, ŷ, y)

Guess the expression of `loss` as a function of prediction and target, by applying it to a
[`PassThroughLayer`](@ref) — a model whose prediction *is* its input.

This is right for every `NetworkLoss` that reaches its model exactly once, as `model(input, ps)`, and
compares the result to `output`, which is what the losses of `AbstractNeuralNetworks` do. It is wrong,
rather than merely unavailable, for a loss that does something else: an autoencoder loss compares the
prediction to the input, and so reads through a pass-through model as identically zero. That is why
the guess is checked — see [`represents_loss`](@ref) — and why [`loss_expression`](@ref) exists.
"""
function passthrough_expression(loss::NetworkLoss, ŷ::AbstractVector{Num}, y::AbstractVector{Num})
    loss(PassThroughLayer{length(ŷ)}(), NetworkParameters(NamedTuple()), ŷ, y)
end

"""
    loss_seed(loss, model; cse, inplace)

Build the seed of the adjoint sweep: a function `(ŷ, y, ps) -> ∂L/∂ŷ`.

Returns `nothing` when the loss cannot be expressed as a function of prediction and target — which is
what makes the layerwise construction fall back to the monolithic one instead of computing the wrong
thing.

The expression comes from [`loss_expression`](@ref) if the loss declares one, and from
[`passthrough_expression`](@ref) otherwise. Only the guess is checked against `loss` itself, with
[`represents_loss`](@ref); a declared expression is used as given.

# Implementation

`ps` is accepted and ignored: the expression has no parameters, and taking one anyway lets the seed be
called exactly like the per-layer kernels of the sweep.
"""
function loss_seed(loss::NetworkLoss, nn::AbstractSymbolicNeuralNetwork;
                   cse::Bool = true, inplace::Bool = true)
    n = output_dimension(nn.model)
    sŷ = Symbolics.variables(:x, 1:n)
    sy = Symbolics.variables(:y, 1:n)
    sparams = NetworkParameters(NamedTuple())

    expression = loss_expression(loss, sŷ, sy)
    if isnothing(expression)
        expression = passthrough_expression(loss, sŷ, sy)
        value = build_nn_function(expression, sparams, sŷ, sy; cse = cse, inplace = inplace)
        represents_loss(loss, nn, value) || return nothing
    end

    build_nn_function(symbolic_derivative(expression, symbolic_differentials(sŷ)), sparams, sŷ, sy;
                      reduce = hcat, cse = cse, inplace = inplace)
end

"""
    represents_loss(loss, nn, value)

Whether the built [`passthrough_expression`](@ref) `value` agrees with `loss` itself, on `nn`'s model,
at a handful of points.

`true` when `loss` cannot be evaluated in the four-argument numeric form at all, in which case there
is nothing to compare against and the expression is taken at its word — that is the case an
overriding method for a loss with its own call signature lands in.

# Implementation

Neither the parameters ([`reference_parameters`](@ref)) nor the points are random: whether a pullback
builds must not depend on the state of the global RNG, and building one must not advance it — a
caller who seeds the RNG and then builds a pullback would otherwise get different data afterwards
than before.

Three points, with different shapes, so that an expression which happens to agree at one of them is
still rejected. The case in mind is an autoencoder loss, which compares the prediction to the input
and so reads as identically zero through a pass-through model — that agrees with the real loss
exactly when the real loss is zero too.
"""
function represents_loss(loss::NetworkLoss, nn::AbstractSymbolicNeuralNetwork, value)
    model = nn.model
    ps = reference_parameters(nn)
    m, n = input_dimension(model), output_dimension(model)

    for scale in (1.0, -0.5, 2.0)
        x = scale .* [cospi(i / 7) for i in 1:m]
        y = [sinpi(scale * i / 5) + 1.5 for i in 1:n]
        prediction = model(x, ps)
        reference = try
            loss(model, ps, x, y)
        catch
            return true
        end
        isapprox(value(prediction, y, ps), reference; rtol = 1e-8) || return false
    end
    true
end

"""
    reference_parameters(nn)

A numeric parameter set with the shape of `nn`'s symbolic one, filled deterministically.

# Implementation

The shape comes from the symbolic parameters through their
`NeuralNetworkParameters.ParameterLayout`, and the numbers from `unflatten`ing a flat vector of them:
a layout built over `Num` leaves unflattens a `Float64` vector into `Float64` leaves, since a leaf is
rebuilt from its prototype's *shape* and not from its element type. Going through the layout rather
than through `initialparameters` is what keeps this free of the global RNG, and it gets structured
parameters right for free.
"""
function reference_parameters(nn::AbstractSymbolicNeuralNetwork)
    layout = parameterlayout(params(nn))
    unflatten(layout, [cospi(i / 11) for i in 1:flatlength(layout)])
end

"""
    LayerStep{Key}(layer, dλ, dθ)

One step of the adjoint sweep: the `layer`, the two generated functions the backward pass calls for
it, and — as a type parameter — the key its parameters have in the parameter set.

Both functions take `(x, λ, ps)`: the layer's own input, the sensitivity of the loss to the layer's
output, and the parameters of the *whole* network. `dλ` returns the sensitivity with respect to the
layer's input, `dθ` the derivative of the loss with respect to the layer's parameters.

Taking the whole parameter set rather than the layer's own entry is what avoids a wrapper per call:
the generated kernels were built from symbolic parameters nested under `Key`, so they reach for
`ps.<Key>.W` and nothing has to be rebuilt for them.

`Key` is a type parameter rather than a field so that reading the layer's own entry out of the
parameter set — [`step_parameters`](@ref), which the forward pass needs — is an inferable
`getproperty` on a name the compiler knows, rather than one on a `Symbol` it does not. Without it a
chain whose layers hold differently shaped parameters infers their union.
"""
struct LayerStep{Key, LT, FT, GT}
    layer::LT
    dλ::FT
    dθ::GT
end

LayerStep{Key}(layer::LT, dλ::FT, dθ::GT) where {Key, LT, FT, GT} =
    LayerStep{Key, LT, FT, GT}(layer, dλ, dθ)

"""
    step_parameters(step, ps)

The entry of the parameter set `ps` that belongs to `step`.
"""
step_parameters(::LayerStep{Key}, ps) where {Key} = ps[Key]

@doc raw"""
    layer_step(layer, key, prototype; cse, inplace)

Build the [`LayerStep`](@ref) of one layer.

# Implementation

Both derivatives come from differentiating the *scalar*

```math
s_k = \lambda_k \cdot f_k(x_{k-1}; \theta_k),
```

in which ``\lambda_k`` is a fresh vector of symbolic variables standing for the sensitivity of the
loss to this layer's output. Its two gradients are precisely what the sweep needs:
``\partial{}s_k/\partial{}x_{k-1}`` **is** ``\lambda_{k-1}`` and ``\partial{}s_k/\partial\theta_k``
**is** ``\partial{}L/\partial\theta_k``.

Seeding the derivative like this rather than building ``\partial{}x_k/\partial{}x_{k-1}`` and
``\partial{}x_k/\partial\theta_k`` separately is what keeps both the expression and the generated code
small: neither the Jacobian nor the rank-3 parameter derivative is ever materialised, and no
contraction is left to do at run time. For four layers of width 16 the seeded form holds 68 760 nodes
against 76 372 for the Jacobian-and-parameter-derivative pair.

The variables at the seam are fresh for every layer, which is the whole point — an expression built
here refers to *this* layer's input and to nothing upstream of it, so its size depends on this layer
alone. That the same names recur across layers is harmless, as each layer is compiled into its own
kernel.

The two derivatives are compiled separately because they are reduced over a batch differently: a
sensitivity is per-sample and concatenates, whereas the gradient of a batch is the sum of the
per-sample gradients — which is what [`SymbolicPullback`](@ref) means by the pullback of a batch. So
the sweep costs two calls per layer whatever the batch size, with no per-sample loop.
"""
function layer_step(layer::AbstractExplicitLayer, key::Symbol, prototype;
                    cse::Bool = true, inplace::Bool = true)
    sx = Symbolics.variables(:x, 1:input_dimension(layer))
    sλ = Symbolics.variables(:λ, 1:output_dimension(layer))
    sparams = NetworkParameters{(key,)}((symbolic_variables(prototype, :W),))

    seed = sum(sλ .* scalar_expressions(layer(sx, sparams[key])))
    differentials = symbolic_differentials(sparams)

    dλ = build_nn_function(symbolic_derivative(seed, symbolic_differentials(sx)), sparams, sx, sλ;
                           reduce = hcat, cse = cse, inplace = inplace)
    dθ = build_nn_function(symbolic_derivative(seed, differentials)[key], sparams, sx, sλ;
                           reduce = +, cse = cse, inplace = inplace)
    LayerStep{key}(layer, dλ, dθ)
end

"""
    symbolic_steps(nn)

The sequence of steps `nn`'s model decomposes into, as `(layer, key)` pairs, or `nothing` when it does
not decompose into one.

A model decomposes when it is a `Chain` whose layers correspond one-to-one to the entries of the
parameter set, and each of whose layers knows the dimensions it maps between — which is what the
layerwise pullback needs in order to put fresh variables at the seams. Anything else takes the
monolithic path.
"""
function symbolic_steps(nn::AbstractSymbolicNeuralNetwork)
    model = nn.model
    model isa Chain || return nothing
    ks = keys(params(nn))
    length(ks) == length(layers(model)) || return nothing
    all(_has_known_dimensions, layers(model)) || return nothing
    ntuple(i -> (layers(model)[i], ks[i]), length(ks))
end

_has_known_dimensions(layer) =
    applicable(input_dimension, layer) && applicable(output_dimension, layer)

@doc raw"""
    composes_layerwise(nn)

Whether composing the pullback layer by layer is the better choice for `nn`, which is what
`layerwise = :auto` asks. True when the model decomposes into *more than one* step.

# Implementation

A single layer is the one case where the monolithic construction wins: there is no composition to
keep out of the expression, so the two build the same derivative, and the seeded form of
[`layer_step`](@ref) merely adds the sensitivity variables and a second generated function. Measured
build times, `FeedForwardLoss` and `Dense` chains, after warm-up:

| layers | width | parameters | layerwise | monolithic |
|-------:|------:|-----------:|----------:|-----------:|
| 1 | 2 | 6 | 0.16 s | **0.06 s** |
| 2 | 4 | 22 | 0.21 s | **0.04 s** |
| 3 | 4 | 42 | 0.29 s | **0.17 s** |
| 4 | 4 | 62 | **0.23 s** | 0.26 s |
| 5 | 4 | 82 | **0.24 s** | 0.40 s |
| 6 | 4 | 102 | **0.26 s** | 0.66 s |
| 4 | 8 | 186 | **0.36 s** | 0.64 s |
| 4 | 16 | 626 | **0.59 s** | does not build |

The layerwise column is flat because it is a sum over layers; the monolithic one grows without bound.
Two and three layers are the rows where the monolithic path is still ahead, by a fraction of a
second — not enough to be worth dispatching on a threshold that would then have to be justified, and
the wrong way to be wrong: the cost of choosing layerwise there is bounded and small, while the cost
of choosing monolithic on a network one layer deeper is unbounded.
"""
function composes_layerwise(nn::AbstractSymbolicNeuralNetwork)
    steps = symbolic_steps(nn)
    !isnothing(steps) && length(steps) > 1
end

"""
    LayerwiseGradientFunction{Keys}(steps, seed)

What a layerwise [`SymbolicPullback`](@ref) stores in place of a single generated gradient function.

Applying it to an input, a target output and the parameters runs the sweep and returns the derivative
of the loss with respect to the parameters, as a `NetworkParameters` — the same thing the monolithic
gradient function returns, so everything above it is unchanged.

`Keys` are the keys of the parameter set, as a type parameter so that the returned `NamedTuple` is
inferable; see [`LayerStep`](@ref) for the same reasoning one level down.
"""
struct LayerwiseGradientFunction{Keys, ST, SDT} <: Function
    steps::ST
    seed::SDT
end

LayerwiseGradientFunction{Keys}(steps::ST, seed::SDT) where {Keys, ST, SDT} =
    LayerwiseGradientFunction{Keys, ST, SDT}(steps, seed)

function (g::LayerwiseGradientFunction{Keys})(input, output, ps) where {Keys}
    NetworkParameters(NamedTuple{Keys}(sweep(g.steps, input, ps, g.seed, output)))
end

"""
    sweep(steps, x, ps, seed, output)

The derivative of the loss with respect to each step's parameters, as a tuple in the order of
`steps`.

# Implementation

The recursion runs the forward pass on the way *in* and the adjoint sweep on the way *out*, so each
intermediate result lives on the stack for exactly as long as the backward pass needs it, and the
whole sweep stays type stable over the tuple of steps.

The first step's `dλ` is never called: it would give the sensitivity of the loss to the network's
*input*, which a parameter gradient has no use for. Hence the two entry points — [`adjoint_step`](@ref)
does the general case, and this function drops that one call.
"""
function sweep(steps::Tuple, x, ps, seed, output)
    step = first(steps)
    y = step.layer(x, step_parameters(step, ps))
    λ, gradients = adjoint_step(Base.tail(steps), y, ps, seed, output)
    (step.dθ(x, λ, ps), gradients...)
end

sweep(::Tuple{}, x, ps, seed, output) = ()

"""
    adjoint_step(steps, x, ps, seed, output)

The sensitivity of the loss to `x` and the parameter gradients of `steps`, given that `x` is what the
remaining `steps` are applied to. See [`sweep`](@ref).
"""
function adjoint_step(steps::Tuple, x, ps, seed, output)
    step = first(steps)
    y = step.layer(x, step_parameters(step, ps))
    λ, gradients = adjoint_step(Base.tail(steps), y, ps, seed, output)
    (step.dλ(x, λ, ps), (step.dθ(x, λ, ps), gradients...))
end

adjoint_step(::Tuple{}, x, ps, seed, output) = (seed(x, output, ps), ())

"""
    layerwise_gradient_function(nn, loss; cse, inplace)

Build the [`LayerwiseGradientFunction`](@ref) of `loss` for `nn`, or `nothing` when the layerwise
construction does not apply — because the model does not decompose into steps
([`symbolic_steps`](@ref)) or because the loss cannot be reduced to a seed ([`loss_seed`](@ref)).
"""
function layerwise_gradient_function(nn::SymbolicNeuralNetwork, loss::NetworkLoss;
                                     cse::Bool = true, inplace::Bool = true)
    steps = symbolic_steps(nn)
    isnothing(steps) && return nothing

    seed = loss_seed(loss, nn; cse = cse, inplace = inplace)
    isnothing(seed) && return nothing

    sparams = params(nn)
    kernels = map(step -> layer_step(step[1], step[2], sparams[step[2]]; cse = cse, inplace = inplace),
                  steps)
    LayerwiseGradientFunction{keys(sparams)}(kernels, seed)
end
