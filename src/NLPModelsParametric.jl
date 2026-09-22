"""
    NLPModelsParametric

Parametric sensitivity interface for `NLPModels`: derivatives with respect to a parameter vector `θ` of length `npar`.

- ``∂c/∂θ``: the constraints Jacobian with respect to the parameters, `ncon × npar`, see [`jac_par`](@ref)
- ``∂²L/∂x∂θ``: the Lagrangian Hessian with respect to the variables and the parameters, `nvar × npar`, see [`hess_par`](@ref)

Sparse coordinate format follows `NLPModels`: `(rows, cols, vals)` with rows in the constraint or variable index and
columns in the parameter index, repeated entries summed.
"""
module NLPModelsParametric

using NLPModels
using SparseArrays

const LAGRANGIAN_HESSIAN_PAR = raw"""
```math
∂²L/∂x∂θ(x,y) = σ ∂²f/∂x∂θ(x) + \sum_i yᵢ ∂²cᵢ/∂x∂θ(x),
```
"""

include("api.jl")

end
