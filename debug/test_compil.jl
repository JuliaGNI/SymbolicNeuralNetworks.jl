using BenchmarkTools

function g(x)
    x^2 * tanh(x) - sin(cos(sin(x)))
end

function f(x)
    (x^2 * tanh(x) - sin(cos(sin(x))))^2 * tanh(x^2 * tanh(x) - sin(cos(sin(x)))) - sin(cos(sin(x^2 * tanh(x) - sin(cos(sin(x))))))
end

function h(x)
    z = g(x)
    g(z)
end

function w(x)
    (g∘g)(x)
end

function zz(x)
    ff(x) = x^2 * tanh(x) - sin(cos(sin(x)))
    a = ff(x)
    ff(a)
end

x = 4
@time f(x)
@time h(x)
@time w(x)
@time zz(x)

using Test
@test f(x) == h(x) == w(x) == zz(x)


