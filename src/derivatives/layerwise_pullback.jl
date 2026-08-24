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
[`checked_guess`](@ref) otherwise. Only the guess is checked against `loss` itself; a declared
expression is used as given.

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
        expression = checked_guess(loss, nn, sŷ, sy; cse = cse, inplace = inplace)
        isnothing(expression) && return nothing
    end

    build_nn_function(symbolic_derivative(expression, symbolic_differentials(sŷ)), sparams, sŷ, sy;
                      reduce = hcat, cse = cse, inplace = inplace)
end

"""
    checked_guess(loss, nn, ŷ, y; cse, inplace)

The guessed expression of `loss` as a function of prediction and target, or `nothing` if the guess
cannot be trusted.

There are two ways for it not to be, and both mean the same thing to the caller — decline, and let
[`SymbolicPullback`](@ref) fall back to the monolithic construction:

- the guess *disagrees* with `loss`, which is what [`represents_loss`](@ref) tests for;
- the guess cannot be **built** at all. A `NetworkLoss` need not accept a
  [`PassThroughLayer`](@ref): the generic four-argument method of `AbstractNeuralNetworks` invites a
  loss to be written for the model it belongs to, and one written as
  `(::MyLoss)(model::Chain, …)` throws when [`passthrough_expression`](@ref) applies it to a model
  that is not a `Chain`. So does a model whose forward pass cannot be evaluated at the points
  [`represents_loss`](@ref) checks at.

The second case has to be caught here rather than left to the caller: `layerwise = :auto` promises a
fallback, and a construction that throws instead of declining would break networks the monolithic
path builds perfectly well.

# Implementation

The `try` covers building and checking the guess, and nothing else. Once an expression is in hand and
has been believed, differentiating it and generating code from it are this package's own work, and a
failure there is a bug to surface rather than a reason to fall back.
"""
function checked_guess(loss::NetworkLoss, nn::AbstractSymbolicNeuralNetwork, ŷ, y;
                       cse::Bool = true, inplace::Bool = true)
    sparams = NetworkParameters(NamedTuple())
    try
        expression = passthrough_expression(loss, ŷ, y)
        value = build_nn_function(expression, sparams, ŷ, y; cse = cse, inplace = inplace)
        represents_loss(loss, nn, value) ? expression : nothing
    catch
        nothing
    end
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

Three points, so that an expression which happens to agree at one of them is still rejected. The case
in mind is an autoencoder loss, which compares the prediction to the input and so reads as identically
zero through a pass-through model — that agrees with the real loss exactly when the real loss is zero
too.

The three differ in *direction* and not merely in scale: three points on one ray through the origin
would leave an expression that is right along that ray and wrong everywhere else undetected, which is
no more work to avoid than to allow.
"""
function represents_loss(loss::NetworkLoss, nn::AbstractSymbolicNeuralNetwork, value)
    model = nn.model
    ps = reference_parameters(nn)
    m, n = input_dimension(model), output_dimension(model)

    for (scale, turn) in ((1.0, 0.0), (-0.5, 1 / 3), (2.0, 2 / 3))
        x = scale .* [cospi(i / 7 + turn) for i in 1:m]
        y = [sinpi(scale * i / 5 + turn) + 1.5 for i in 1:n]
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

A layer that carries data alongside the state takes that data in between, as further arguments before
`λ`; [`seam_arguments`](@ref) is what produces them, and [`seam_interface`](@ref) the whole picture.

`dλ` is `nothing` for the first step of a sweep, which is the one place it is never called; see
[`layer_step`](@ref).

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
    layer_step(layer, key, seeded; input_sensitivity, cse, inplace)

Build the [`LayerStep`](@ref) of one layer from `seeded`, the tuple [`layer_seed`](@ref) returns.

The seed is passed in rather than built here so that every layer of a chain can be seeded — and the
chain declined if one of them cannot be, see [`checked_layer_seed`](@ref) — *before* any code is
generated for any of them. Building a seed is one symbolic forward pass through one layer; generating
its two kernels is the expensive half, and a chain that will be declined should not pay it.

`input_sensitivity = false` leaves out the derivative with respect to the layer's *input*, for the
first layer of a chain, whose sensitivity is that of the loss to the network's input — something a
parameter gradient has no use for, and which [`sweep`](@ref) accordingly never asks for. Generating
it anyway would be half the code this function emits, spent on a function that is never called.

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
function layer_step(layer::AbstractExplicitLayer, key::Symbol, seeded::Tuple;
                    input_sensitivity::Bool = true, cse::Bool = true, inplace::Bool = true)
    seed, sparams, sdata, sλ = seeded
    # the state is the first of the seam's data variables, and the only one differentiated with
    # respect to: what a layer carries alongside it is data — see `seam_interface`
    sx = first(sdata)

    dλ = input_sensitivity ?
         build_nn_function(symbolic_derivative(seed, symbolic_differentials(sx)), sparams, sdata...,
                           sλ; reduce = hcat, cse = cse, inplace = inplace) : nothing
    dθ = build_nn_function(symbolic_derivative(seed, symbolic_differentials(sparams[key])), sparams,
                           sdata..., sλ; reduce = +, cse = cse, inplace = inplace)
    LayerStep{key}(layer, dλ, dθ)
end

@doc raw"""
The seam interface
------------------

`layer_seed` puts *fresh* symbolic variables between two layers, which is what keeps the symbolic
material a sum over layers rather than a product. By default those variables are one plain vector —
the layer's state — because that is what a `Dense` maps to a `Dense`.

A layer may carry more than the state. `GeometricMachineLearning`'s `SymplecticEuler` threads the
parameters of the *system* through the chain alongside it, so it takes and returns a pair. Four
functions say how such a layer meets the seam; each defaults to exactly the plain-vector construction,
so a layer that carries nothing needs none of them, and a layer that carries something declares all
four together:

| function | what it answers | default |
|---|---|---|
| [`carried_variables`](@ref) | what fresh variables the carried data needs | `()` |
| [`seam_value`](@ref) | what the layer is *applied to* at the seam | the state alone |
| [`state_expressions`](@ref) | which part of the output ``\lambda`` pairs with | all of it |
| [`seam_arguments`](@ref) | the *run-time* arguments of the generated kernels | the input alone |

The carried data is **data, never a differentiation target**: ``\lambda`` pairs with the state, the
seed is differentiated with respect to the state and with respect to the layer's parameters, and the
carried variables are extra arguments of the generated kernels. So a layer that carries something gets
the same two kernels as any other, taking one more argument each.

`seam_arguments` must return arrays with the *same rank and batch size* as the state, in the order
`(state, carried…)` that `carried_variables` declared — the constraint every generated function of
this package imposes on its data arguments. For carried data that is the same for the whole batch that
means broadcasting it out to one column per sample.
"""
function seam_interface end

"""
    carried_variables(layer)

Fresh symbolic arrays standing for whatever `layer` carries alongside the state at the seam, as a
tuple. `()` by default, which is the plain-vector seam.

# Extending

```julia
SymbolicNeuralNetworks.carried_variables(layer::MyLayer) = (Symbolics.variables(:c, 1:length(layer)),)
```

The arrays are built here rather than passed in because their *shape* is the layer's own knowledge —
this package knows only `input_dimension` and `output_dimension`, which describe the state.

Return `()` when there is nothing to carry, rather than an empty array: there would be nothing to hand
the generated kernels, and an empty array is not a usable data argument in any case. The seam is then
the plain vector it is for every other layer. `SymplecticEuler` over a system with no parameters is
exactly this case.

The layer may still have to be *given* something — its functor takes the pair either way — in which
case [`seam_value`](@ref) supplies it as a constant. Nothing varies, so nothing needs a variable, and
the constant is folded into the expression like any other. See [`seam_interface`](@ref).
"""
carried_variables(::AbstractExplicitLayer) = ()

"""
    seam_value(layer, sx, carried...)

What `layer` is applied to at the seam, assembled from the state variables `sx` and the arrays
[`carried_variables`](@ref) returned. `sx` alone by default.

# Extending

```julia
SymbolicNeuralNetworks.seam_value(layer::MyLayer, sx, sc) = (sx, sc)
```

A layer whose carried data reaches it in some other shape than the flat array the kernels take —
`SymplecticEuler` wants a `NamedTuple` of system parameters — reassembles it here, and takes it apart
again in [`seam_arguments`](@ref). See [`seam_interface`](@ref).
"""
seam_value(::AbstractExplicitLayer, sx, carried...) = sx

@doc raw"""
    state_expressions(layer, y)

The part of `layer`'s output ``\lambda`` pairs with, as scalar expressions. All of it by default.

# Extending

```julia
SymbolicNeuralNetworks.state_expressions(::MyLayer, y) = scalar_expressions(first(y))
```

A layer that passes its carried data on returns it beside the state, and the seed is
``\lambda_k\cdot{}f_k(x_{k-1};\theta_k)`` over the *state* — so this is what says which part that is.
See [`seam_interface`](@ref).
"""
state_expressions(::AbstractExplicitLayer, y) = scalar_expressions(y)

"""
    seam_arguments(layer, x)

The data arguments the kernels generated for `layer` are called with, given the layer's run-time
input `x`: a tuple in the order `(state, carried...)` that [`carried_variables`](@ref) declared.
`(x,)` by default.

# Extending

```julia
SymbolicNeuralNetworks.seam_arguments(::MyLayer, x::Tuple) = (first(x), flat(last(x)))
```

This is the run-time counterpart of [`seam_value`](@ref), and the two have to agree: what the seam is
*written in* and what it is *called with* are the same list. Every entry must have the same rank and
batch size as the state. See [`seam_interface`](@ref).
"""
seam_arguments(::AbstractExplicitLayer, x) = (x,)

@doc raw"""
    layer_seed(layer, key, prototype)

The scalar one layer's two derivatives are taken of, together with the variables it is written in:
`(seed, sparams, sdata, sλ)`, where

```math
\mathrm{seed} = \lambda_k \cdot f_k(x_{k-1}; \theta_k).
```

`sparams` nests the layer's parameters under `key`, with the shape of `prototype`, so that the code
generated from `seed` reaches into the parameter set of the *whole* network — see
[`LayerStep`](@ref). `sdata` is the tuple of data variables the seam is written in — the state first,
then whatever [`carried_variables`](@ref) declared — and it and `sλ` are fresh for every layer, which
is what keeps an expression built from this one dependent on that layer alone.

A layer that carries nothing gets `sdata = (sx,)`, one plain vector, and this is then the construction
it always was. A layer that carries something and has *not* declared the seam interface cannot be
seeded at all — it either returns more than `state_expressions`' default can take apart, or has no
method for a bare vector in the first place; [`checked_layer_seed`](@ref) is what turns that into a
decline rather than an exception. See [`seam_interface`](@ref).

Separate from [`layer_step`](@ref) so that `scripts/codegen_comparison.jl` measures the symbolic
material this construction actually holds, rather than a second copy of it that can drift.
"""
function layer_seed(layer::AbstractExplicitLayer, key::Symbol, prototype)
    sx = Symbolics.variables(:x, 1:input_dimension(layer))
    sλ = Symbolics.variables(:λ, 1:output_dimension(layer))
    sdata = (sx, carried_variables(layer)...)
    sparams = NetworkParameters{(key,)}((symbolic_variables(prototype, :W),))
    value = layer(seam_value(layer, sdata...), sparams[key])
    (sum(sλ .* state_expressions(layer, value)), sparams, sdata, sλ)
end

"""
    checked_layer_seed(layer, key, prototype)

What [`layer_seed`](@ref) returns for `layer`, or `nothing` when the layer cannot be seeded at all.

A layer that carries something alongside the state and has not declared the seam interface (see
[`seam_interface`](@ref)) has no seed, because the seam it is offered is a plain vector of symbolic
variables. There are two ways for that to show, and both mean the same thing to the caller — decline,
and let [`SymbolicPullback`](@ref) fall back to the monolithic construction:

- the layer *returns* more than the state, so [`state_expressions`](@ref)' default has no method for
  its output. This is `GeometricMachineLearning`'s `SymplecticEuler` with `return_parameters = true`,
  which threads the parameters of the system on to the next layer and returns a `Tuple`;
- the layer cannot be *applied* to the bare vector at the seam in the first place, which is what the
  layer downstream of such a one does — its input is the tuple, and it has no method for anything
  else.

The second is why this catches rather than asking `applicable(scalar_expressions, layer(sx, ps))`:
that predicate needs the layer to have been applied already, so it only covers the first.

# Implementation

The `try` covers building the seed and nothing else — the same line [`checked_guess`](@ref) draws one
level up. Once a seed is in hand, differentiating it and generating code from it are this package's
own work, and a failure there is a bug to surface rather than a reason to fall back.
"""
function checked_layer_seed(layer::AbstractExplicitLayer, key::Symbol, prototype)
    try
        layer_seed(layer, key, prototype)
    catch
        nothing
    end
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
    ls = layers(model)
    length(ks) == length(ls) || return nothing
    all(_has_known_dimensions, ls) || return nothing
    ntuple(i -> (ls[i], ks[i]), length(ks))
end

_has_known_dimensions(layer) =
    applicable(input_dimension, layer) && applicable(output_dimension, layer)

@doc raw"""
    composes_layerwise(nn)

Whether composing the pullback layer by layer is the better choice for `nn`, which is what
`layerwise = :auto` asks. True when the model decomposes into *more than one* step.

This is a question about which construction is *preferable*, not about whether the layerwise one
applies: the layers still have to be seedable ([`checked_layer_seed`](@ref)) and the loss still has to
reduce to a seed ([`loss_seed`](@ref)), both of which are settled afterwards by
[`layerwise_gradient_function`](@ref). So a `true` here can still be followed by a decline.

# Implementation

A single layer is the one case where the monolithic construction is unambiguously right: there is no
composition to keep out of the expression, so the two build the same derivative, and the seeded form
of [`layer_step`](@ref) merely adds the sensitivity variables and a second generated function.

Above one layer the two are measured against each other by `scripts/codegen_comparison.jl`. In
expression nodes — the size of the symbolic material each construction has to hold — and in seconds
to build, for `Dense` chains with a `FeedForwardLoss`:

| layers | width | parameters | monolithic nodes | layerwise nodes | monolithic | layerwise |
|-------:|------:|-----------:|-----------------:|----------------:|-----------:|----------:|
| 2 | 4 | 22 | 6 652 | 792 | **0.02 s** | 0.20 s |
| 3 | 4 | 42 | 57 772 | 1 656 | **0.04 s** | 0.23 s |
| 4 | 4 | 62 | 388 700 | 2 520 | **0.10 s** | 0.21 s |
| 5 | 4 | 82 | 2 317 964 | 3 384 | 0.34 s | **0.30 s** |
| 6 | 4 | 102 | 12 848 828 | 4 248 | 0.65 s | **0.32 s** |
| 4 | 8 | 186 | 8 253 148 | 11 736 | 0.61 s | **0.30 s** |
| 4 | 16 | 626 | 209 455 964 | 68 760 | does not build | **0.53 s** |

The layerwise node count is exactly linear — 864 more per identical added layer — against a column
that multiplies by about six each time.

Build time crosses over at five layers, or at four of width eight. Below that the monolithic path is
still ahead, by at most a fifth of a second, and this returns `true` there anyway: a threshold that
reproduced the crossover would have to be a function of width as well as depth, fitted to timings
from one machine, and it would be the wrong thing to get wrong. Choosing layerwise where monolithic
would have been quicker costs a fixed fraction of a second; choosing monolithic on a network one
layer deeper costs everything.
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
    NetworkParameters(NamedTuple{Keys}(sweep(g.steps, batched(input), ps, g.seed, batched(output))))
end

@doc raw"""
    batched(data)

`data` with any batch dimensions past the first collapsed into it, which is the shape the sweep
evaluates in. For an input that carries data alongside the state, only the state is laid out; the rest
is [`seam_arguments`](@ref)' business.

# Implementation

[`build_nn_function`](@ref) accepts a result of ``m\times{}N_1\times{}N_2`` as a batch with two
batch dimensions (see [`AbstractBatchedFunction`](@ref)), and the monolithic construction of
[`SymbolicPullback`](@ref) therefore evaluates one. The layerwise sweep cannot pass such an array to a
layer — the forward pass is the layer *called*, and a `Dense` multiplies a matrix by a matrix — so the
batch is laid out flat first.

That loses nothing. The pullback of a batch is the *sum* of the per-sample gradients, so how the
samples are arranged cannot change it; the two shapes are checked against each other in
`test/derivatives/layerwise_pullback.jl`. Nothing has to be restored afterwards either, for the same
reason: what comes back is shaped like the parameters, not like the batch.
"""
batched(data::AbstractVector) = data
batched(data::AbstractMatrix) = data
batched(data::AbstractArray) = reshape(data, size(data, 1), :)
# an input that carries data alongside the state — see `seam_interface` — has the state in its first
# entry, and it is only the state that a layer's forward pass has to be able to multiply by
batched(data::Tuple) = (batched(first(data)), Base.tail(data)...)

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
does the general case, and this function drops that one call. The first step is not built with a `dλ`
at all, so dropping it here is what makes that legal as well as cheaper; see [`layer_step`](@ref).
"""
function sweep(steps::Tuple, x, ps, seed, output)
    step = first(steps)
    y = step.layer(x, step_parameters(step, ps))
    λ, gradients = adjoint_step(Base.tail(steps), y, ps, seed, output)
    (step.dθ(seam_arguments(step.layer, x)..., λ, ps), gradients...)
end

sweep(::Tuple{}, x, ps, seed, output) = ()

"""
    adjoint_step(steps, x, ps, seed, output)

The sensitivity of the loss to `x` and the parameter gradients of `steps`, given that `x` is what the
remaining `steps` are applied to. See [`sweep`](@ref).

The recursion bottoms out at the network's *output*, which the seed takes together with the target. So
the chain's last layer has to return the model's output and nothing beside it — a layer that carries
data through cannot be the last one, which is also what the monolithic construction and the loss
itself require.
"""
function adjoint_step(steps::Tuple, x, ps, seed, output)
    step = first(steps)
    y = step.layer(x, step_parameters(step, ps))
    λ, gradients = adjoint_step(Base.tail(steps), y, ps, seed, output)
    # `seam_arguments` returns a tuple whose length is fixed per layer type, so this stays inferable
    arguments = seam_arguments(step.layer, x)
    (step.dλ(arguments..., λ, ps), (step.dθ(arguments..., λ, ps), gradients...))
end

adjoint_step(::Tuple{}, x, ps, seed, output) = (seed(x, output, ps), ())

"""
    decline(demanded, why)

Refuse the layerwise construction, `why` saying what stood in the way.

Returns `nothing` — the signal `layerwise = :auto` falls back to the monolithic construction on — or
throws an `ArgumentError` naming `why`, when the caller asked for the construction by name with
`layerwise = true`.

# Implementation

Both outcomes go through here so that [`layerwise_gradient_function`](@ref) has *one* decision path:
the message is raised where the decline happens rather than reconstructed afterwards by a second
traversal of the same checks, which could not help but drift from them.
"""
function decline(demanded::Bool, why::AbstractString)
    demanded && throw(ArgumentError(
        "`layerwise = true`, but the pullback cannot be built layer by layer for this network: " *
        why * ". Pass `layerwise = :auto` to fall back to the monolithic construction."))
    nothing
end

"""
    unseedable_reason(steps, seeds)

The `why` [`decline`](@ref) is given when one of a chain's layers cannot be seeded: the keys and types
of every layer whose entry in `seeds` came back `nothing`.
"""
function unseedable_reason(steps::Tuple, seeds::Tuple)
    named = join(("`$(steps[i][2])` (`$(nameof(typeof(steps[i][1])))`)"
                  for i in eachindex(seeds) if isnothing(seeds[i])), ", ", " and ")
    "the layers " * named * " cannot be seeded, because the seam the construction puts between two " *
    "layers is a plain vector of symbolic variables: a layer has to map an array to an array and " *
    "carry nothing alongside the state (see `layer_seed` and `checked_layer_seed`)"
end

"""
    layerwise_gradient_function(nn, loss; demanded, cse, inplace)

Build the [`LayerwiseGradientFunction`](@ref) of `loss` for `nn`, or [`decline`](@ref) when the
layerwise construction does not apply. There are three ways for it not to:

- the model does not decompose into steps ([`symbolic_steps`](@ref));
- one of those steps cannot be seeded ([`checked_layer_seed`](@ref));
- the loss cannot be reduced to a seed ([`loss_seed`](@ref)).

`demanded = true` — which is `SymbolicPullback`'s `layerwise = true` — makes each of those an error
naming the reason instead of the `nothing` that falls back.

# Implementation

The layers are checked before the loss because the check is cheaper: [`loss_seed`](@ref) builds a
function, evaluates it at three points and then differentiates and builds again, whereas seeding a
layer is one symbolic forward pass. So a chain that cannot be seeded declines before any code is
generated at all. The order also decides which reason a model that declines on both counts reports.
"""
function layerwise_gradient_function(nn::SymbolicNeuralNetwork, loss::NetworkLoss;
                                     demanded::Bool = false, cse::Bool = true, inplace::Bool = true)
    steps = symbolic_steps(nn)
    isnothing(steps) && return decline(demanded,
        "the model does not decompose into a sequence of layers with known dimensions " *
        "(see `symbolic_steps`)")

    sparams = params(nn)
    seeds = ntuple(length(steps)) do i
        layer, key = steps[i]
        checked_layer_seed(layer, key, sparams[key])
    end
    any(isnothing, seeds) && return decline(demanded, unseedable_reason(steps, seeds))

    seed = loss_seed(loss, nn; cse = cse, inplace = inplace)
    isnothing(seed) && return decline(demanded,
        "the loss cannot be expressed as a function of the prediction and the target " *
        "(see `loss_expression`)")

    # `input_sensitivity = i > 1`: the sweep never asks the first layer for the sensitivity to its
    # input, so that derivative is not generated for it
    kernels = ntuple(length(steps)) do i
        layer, key = steps[i]
        layer_step(layer, key, seeds[i]; input_sensitivity = i > 1, cse = cse, inplace = inplace)
    end
    LayerwiseGradientFunction{keys(sparams)}(kernels, seed)
end
