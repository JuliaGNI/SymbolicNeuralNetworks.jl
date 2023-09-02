############################################
#           Abstract Class
############################################

abstract type AbstractSymbolFunction{S} end

abstract type OperationSymbolFunction{S} <: AbstractSymbolFunction{S} end

struct SymbolFunction{S} <: AbstractSymbolFunction{S}
    SymbolFunction(s::Symbol) = new{s}()
end

TypeDim = Tuple

struct TensorSymbolicFunction{TM,TN,TExpr}
    dimIn::TypeDim
    dimOut::TypeDim
    expr::TExpr
    dleft::String
    dright::String
    function TensorSymbolicFunction(f::AbstractSymbolFunction{S}, dimIn::TypeDim, dimOut::TypeDim, dleft = "", dright ="") where S
        new{length(dimIn),length(dimOut),typeof(f)}(dimIn,dimOut,f,dleft,dright)
    end
end

laxes(::TensorSymbolicFunction{TM,TN}) where {TM,TN} = (TM,TN)
type(::TensorSymbolicFunction{TM,TN,TExpr}) where {TM,TN,TExpr} = TExpr
symbol(::TensorSymbolicFunction{TM,TN,<:AbstractSymbolFunction{S}}) where {TM,TN,S} = S
dimIn(tsf::TensorSymbolicFunction) = tsf.dimIn
dimOut(tsf::TensorSymbolicFunction) = tsf.dimOut
expr(tsf::TensorSymbolicFunction) = tsf.expr
dlim(tsf::TensorSymbolicFunction) = (tsf.dleft, tsf.dright)
args(tsf::TensorSymbolicFunction) = args(tsf.expr)

TensorSymbolicFunction(f::Symbol, dimIn::TypeDim, dimOut::TypeDim, dleft = "", dright ="") = TensorSymbolicFunction(SymbolFunction(f),dimIn,dimOut,dleft,dright)

TSF(f::Union{AbstractSymbolFunction,Symbol}, dimIn::Union{Int,TypeDim}, dimOut::Union{Int,TypeDim}, dleft = "", dright ="") = TensorSymbolicFunction(f, TypeDim(dimIn), TypeDim(dimOut), dleft, dright)


function Base.show(io::IO, f::TensorSymbolicFunction)
    print(io, string(dlim(f)[1],symbol(f),dlim(f)[2]))
end

const VectorSymbolicFunction{TExpr}  = TensorSymbolicFunction{1,1,TExpr} where {TExpr}
const MaxtrixSymbolicFunction{TExpr} = TensorSymbolicFunction{1,2,TExpr} where {TExpr}

############################################
#           IndexSymbolFunction
############################################

typeIndex = Union{Int,Symbol}

projectToFirst(t::Tuple, n::Int) = t[1+n:end]

struct IndexSymbolFunction{S,N} <: AbstractSymbolFunction{S}
    f::TensorSymbolicFunction
    index
    function IndexSymbolFunction(f::TensorSymbolicFunction{TM,TN,TE}, index::typeIndex...) where{TM,TN,TE}
        @assert length(index) <= TN "Too many index."
        for (i,n) in zip(index,1:length(index))
            if typeof(i) != Symbol
                @assert i<=dimOut(f)[n] string(n,"ᵗʰ index have a dimension of ", dimOut(f)[n],".")
            end
        end
        str_s = TN == 0 ? string(symbol(f)) : string(symbol(f),"[",string([string(i)*"," for i in index]...)[1:end-1],"]")
        new{Symbol(str_s),length(index)}(f,index)
    end
end

function Base.getindex(f::TensorSymbolicFunction{TM,TN}, index::typeIndex...) where {TM,TN}
    if TN == 0
        return f
    end
    TSF(IndexSymbolFunction(f,index...),dimIn(f),projectToFirst(dimOut(f),length(index)))
end

function Base.getindex(f::TensorSymbolicFunction{TM,TN,<:IndexSymbolFunction}, index::typeIndex...) where{TM,TN}
    new_expr = IndexSymbolFunction(f.expr.f,(f.expr.index...,index...)...)
    TSF(new_expr,dimIn(f),projectToFirst(dimOut(f),length(index)),dlim(f)...)
end

args(sf::IndexSymbolFunction) = (sf.f,sf.index)

############################################
#               Addition
############################################

struct AddSymbolFunction{S} <: OperationSymbolFunction{S}
    left::TensorSymbolicFunction
    right::TensorSymbolicFunction
    function AddSymbolFunction(f::TensorSymbolicFunction{TM,TN},g::TensorSymbolicFunction{TQ,TP,TE}) where {TM,TN,TQ,TP,TE}
        @assert TM ==  TP  string("You cannot add two functions with different ouput spaces.")
        @assert TM ==  TQ  string("You cannot add two functions with different input spaces.")
        str_op = TE <: MinusSymbolFunction ? "-" : "+"
        new{Symbol(string(symbol(f),str_op,symbol(g)))}(f,g)
    end
end

function Base.:+(f::TensorSymbolicFunction, g::TensorSymbolicFunction)
    TSF(AddSymbolFunction(f,g),dimIn(f),dimOut(f))
end

args(sf::AddSymbolFunction) = (sf.left,sf.right)

############################################
#                   Minus
############################################

struct MinusSymbolFunction{S} <: AbstractSymbolFunction{S}
    function MinusSymbolFunction(f::TensorSymbolicFunction)
        new{symbol(f)}()
    end
end

function Base.:-(f::TensorSymbolicFunction)
    TSF(MinusSymbolFunction(f),dimIn(f),dimOut(f),"-")
end

function Base.:-(f::TensorSymbolicFunction, g::TensorSymbolicFunction)
    TSF(AddSymbolFunction(f, -g),dimIn(f),dimOut(f))
end



############################################
#               Derivative
############################################

abstract type Derivative{D} end

symbol(::Derivative{D}) where D = D


struct DerivativeSymbolFunction{S} <: AbstractSymbolFunction{S}
    D::Derivative
    f::AbstractSymbolFunction
    function DerivativeSymbolFunction(d::Derivative{D}, f::TensorSymbolicFunction) where D
        new{Symbol(string(D,symbol(f)))}(d,f)
    end
end

struct PartialDerivative{D, I} <: Derivative{D}
    PartialDerivative(i::Int) = new{Symbol(string("∂[",i,"]")),i}()
end

struct Gradient{D,N} <: Derivative{D} 
    Gradient(n::Int=1) = new{Symbol(string("∇",repeat("⊗ ∇",n-1))),n}()
end

const JacobianSymbol{D} = Gradient{D,2}

