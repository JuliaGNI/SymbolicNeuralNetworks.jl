function parallelize_expression_inplace(expression!::Base.Callable)
    @kernel function parallellize_expression_inplace_kernel!(output::AT, input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
        j, k = @index(Global, NTuple)
    
        expression!(output[:, j, k], input[:, j, k], ps)
        nothing
    end

    function parallelized_expression_inplace(output::AT, input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
        backend = KernelAbstractions.get_backend(output)
        kernel! = parallellize_expression_inplace_kernel!(backend)
        kernel!(output, input, ps, ndrange=(size(output, 2), size(output, 3)))
        nothing
    end

    parallelized_expression_inplace
end

function parallelize_expression(expression::Base.Callable)
    @kernel function parallellize_expression_kernel!(output::AT, input::AT, ps, expression::Base.Callable) where {T, AT <: AbstractArray{T, 3}}
        j, k = @index(Global, NTuple)
    
        output[:, j, k] .= expression(input[:, j, k], ps)
        nothing
    end

    function parallelized_expression(input::AT, ps) where {T, AT <: AbstractArray{T, 3}}
        backend = KernelAbstractions.get_backend(output)
        output = KernelAbstractions.zeros(backend, T, size(input)...)
        kernel! = parallellize_expression_kernel!(backend)
        kernel!(output, input, ps, ndrange=(size(output, 2), size(output, 3)))
        output
    end

    parallelized_expression
end