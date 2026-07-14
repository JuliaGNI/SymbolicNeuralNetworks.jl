# Runs the doctests embedded in the package's docstrings.
#
# The authoritative doctest run happens in the documentation build (see
# `.github/workflows/Documenter.yml`), which is why this is *not* part of the default
# `runtests.jl` set: doctest output is sensitive to the Julia/Symbolics versions, and the
# CI test matrix spans several of them. `runtests.jl` includes this file only when the
# `SYMBOLICNEURALNETWORKS_DOCTESTS` environment variable is set, so developers can opt in
# with `SYMBOLICNEURALNETWORKS_DOCTESTS=true julia --project=docs test/doctest.jl` (or via
# `Pkg.test()` with the same variable set).

using SymbolicNeuralNetworks
using Documenter: DocMeta, doctest

DocMeta.setdocmeta!(SymbolicNeuralNetworks, :DocTestSetup, :(using SymbolicNeuralNetworks); recursive = true)

doctest(SymbolicNeuralNetworks)
