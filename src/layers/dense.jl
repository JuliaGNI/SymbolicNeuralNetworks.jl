function symbolicparameters(::Dense{M, N, true}) where {M,N}
    @variables W[1:N, 1:M], b[1:N]
    (W = W, b = b)
end

function symbolicparameters(::Dense{M, N, false}) where {M,N}
    @variables W[1:N, 1:M]
    (W = W,)
end