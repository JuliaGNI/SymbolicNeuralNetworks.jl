using Symbolics

@variables x
@variables y
@variables n(x,y)
Dy = Differential(y)
Dx = Differential(x)
eq = Dx(Dy(n))

eqs = (x = x, y = y, nn = n, eq = eq)


function symbolise(eqs)

    @show inputx = eqs.x
    @show inputy = eqs.y
    @show nn = eqs.nn
    @show eq = eqs.eq

    f(x, y) = x^2 + 2*x*y^2 + x*y

    @show sf = f(inputx, inputy)

    expand_derivatives(SymbolicUtils.substitute(eq, [nn=>sf]))

end

symbolise(eqs)