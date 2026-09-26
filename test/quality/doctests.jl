# The doctests of this package, in its docstrings and in the manual under `docs/src`, as the
# `Doctests` job of `CI.yml` runs them.
#
# Documenter evaluates a page's `@meta` block in `Main`, and a `@safetestset` file runs in a module
# of its own, so `SymbolicNeuralNetworks` is imported into `Main` first.

using SymbolicNeuralNetworks
using Documenter: DocMeta, doctest

@eval Main import SymbolicNeuralNetworks

DocMeta.setdocmeta!(SymbolicNeuralNetworks, :DocTestSetup, :(using SymbolicNeuralNetworks); recursive = true)

doctest(SymbolicNeuralNetworks)
