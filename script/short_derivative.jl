############################################
#           Utils for Symbol
############################################

function add_brace_extr(s::Symbol)
    str = string(s) 
    if str[begin] != '(' && str[end] != ')' 
        return Symbol(string("(",s,")"))
    end
    s
end

function remove_brace_extr(s::Symbol)
    str = string(s) 
    if str[begin] == '(' && str[end] == ')'
        return Symbol(str[begin+1:end-1])
    end
    s
end

function add_minus_extr(s::Symbol)
    str = string(s) 
    if str[begin] != '-'
        return Symbol(string("-",str))
    end
    s
end

############################################
#           Abstract Class
############################################

abstract type AbstractTensorSymbolicFunction{M,N} end

const ATSF = AbstractTensorSymbolicFunction

abstract type OperationSymbolicFunction{M,N} <: AbstractTensorSymbolicFunction{M,N} end

dim_input(::AbstractTensorSymbolicFunction{M,N}) where {M,N} = M
dim_output(::AbstractTensorSymbolicFunction{M,N}) where {M,N} = N
symbol(f::AbstractTensorSymbolicFunction) = f.s

function Base.show(io::IO, f::AbstractTensorSymbolicFunction{M,N}) where {M,N}
    print(io, symbol(f))
end



############################################
#           basic Function
############################################

struct TensorSymbolicFunction{M,N} <: AbstractTensorSymbolicFunction{M,N}
    s::Symbol
    TensorSymbolicFunction(s::Symbol,m::Int, n::Int) = new{m,n}(s)
end

const ScalarSymbolicFunction = TensorSymbolicFunction{M,0} where M
const VectorSymbolicFunction  = TensorSymbolicFunction{M,1} where M
const MaxtrixSymbolicFunction = TensorSymbolicFunction{M,2} where M

TensorSF(s::Symbol, m::Int, n::Int) = TensorSymbolicFunction(s,m,n)
ScalarSF(s::Symbol, m::Int = 0) = TensorSymbolicFunction(s,m,0)
VectorSF(s::Symbol, m::Int = 1) = TensorSymbolicFunction(s,m,1)
MatrixSF(s::Symbol, m::Int = 1) = TensorSymbolicFunction(s,m,2)


############################################
#               Addition
############################################

struct AddSymbolicFunction{M,N} <: OperationSymbolicFunction{M,N}
    s::Symbol
    left::AbstractTensorSymbolicFunction
    right::AbstractTensorSymbolicFunction
    function AddSymbolicFunction(f::AbstractTensorSymbolicFunction{M,N},g::AbstractTensorSymbolicFunction{Q,P}) where {M,N,Q,P}
        @assert N ==  P  string("You cannot add two functions with different ouput spaces.")
        @assert M ==  Q  string("You cannot add two functions with different input spaces.")
        str_op = typeof(g) <: MinusSymbolicFunction ? "" : "+"
        new{M,N}(Symbol(string(symbol(f), str_op, symbol(g))),f,g)
    end
end

function Base.:+(f::AbstractTensorSymbolicFunction, g::AbstractTensorSymbolicFunction)
    AddSymbolicFunction(f,g)
end

args(sf::AddSymbolicFunction) = (sf.left,sf.right)

############################################
#                  Minus
############################################

struct MinusSymbolicFunction{M,N} <: AbstractTensorSymbolicFunction{M,N}
    s::Symbol
    function MinusSymbolicFunction(f::AbstractTensorSymbolicFunction{M,N}) where {M,N}
        str_f = typeof(f) <: TensorSymbolicFunction ? symbol(f) : add_brace_extr(symbol(f))
        str_f = add_minus_extr(str_f)
        new{M,N}(str_f)
    end
end

function Base.:-(f::AbstractTensorSymbolicFunction)
    MinusSymbolicFunction(f)
end

function Base.:-(f::AbstractTensorSymbolicFunction, g::AbstractTensorSymbolicFunction)
    AddSymbolicFunction(f, MinusSymbolicFunction(g))
end

############################################
#               Multiplication
############################################

struct MultSymbolicFunction{M,N} <: OperationSymbolicFunction{M,N}
    s::Symbol
    left::AbstractSymbolicFunction
    right::AbstractSymbolFunction
    function MultSymbolicFunction(f::AbstractTensorSymbolicFunction{M,N},g::AbstractTensorSymbolicFunction{Q,P}) where {M,N,Q,P}
        @assert M == Q string("You cannot multiply two functionswith different input.")
        @assert P ∈{N-1,N} string("You cannot multiply a tensor of order ",N," with a tensor of order ",Q,".")
        str_f = typeof(f)<:MultSymbolicFunction ? remove_brace_extr(symbol(f)) : f
        str_g = typeof(g)<:MultSymbolicFunction ? remove_brace_extr(symbol(g)) : g
        new{M,P}(Symbol(string("(",str_f,*,str_g,")")),f,g)
    end
end

function Base.:*(f::AbstractTensorSymbolicFunction, g::AbstractTensorSymbolicFunction)
    MultSymbolicFunction(f,g)
end

############################################
#               PointwiseProduct
############################################


############################################
#               Composition
############################################

struct ComposedSymbolicFunction{M,N} <: OperationSymbolFunction{M,N}
    s::Symbol
    outer::AbstractSymbolFunction
    inner::AbstractSymbolFunction
    function ComposedSymbolicFunction(f::AbstractSymbolFunction{M,N},g::AbstractSymbolFunction{Q,P}) where {M,N,Q,P}
        @assert N == Q string("Dimension of output of ",f," must match with the dimension of intput of ",g,".")
        str_f = typeof(f)<:ComposedSymbolFunction ? remove_brace_extr(symbol(f)) : symbol(f)
        str_g = typeof(g)<:ComposedSymbolFunction ? remove_brace_extr(symbol(g)) : symbol(g)
        new{M,P,Symbol(string("(",str_g,∘,str_f,")"))}(g,f)
    end
end

function Base.ComposedFunction(g::AbstractSymbolFunction{Q,P}, f::AbstractSymbolFunction{M,N}) where {M,N,Q,P}
    ComposedSymbolFunction(f,g)
end

function Base.ComposedFunction(g::MultSymbolFunction, f::AbstractSymbolFunction)
    MultSymbolFunction(ComposedSymbolFunction(f,g.left),ComposedSymbolFunction(f,g.right))
end

function Base.ComposedFunction(g::AddSymbolFunction, f::AbstractSymbolFunction)
    AddSymbolFunction(ComposedSymbolFunction(f,g.left),ComposedSymbolFunction(f,g.right))
end
