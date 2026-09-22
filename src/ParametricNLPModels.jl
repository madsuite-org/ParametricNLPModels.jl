"""
    ParametricNLPModels

Parametric sensitivity interface for `NLPModels`: derivatives with respect to a parameter vector `θ` of length `npar`.

- ``∂f/∂θ``: the objective gradient with respect to the parameters, `npar`, see [`grad_par`](@ref)
- ``∂c/∂θ``: the constraints Jacobian with respect to the parameters, `ncon × npar`, see [`jac_par`](@ref)
- ``∂²L/∂x∂θ``: the Lagrangian Hessian with respect to the variables and the parameters, `nvar × npar`, see [`hess_par`](@ref)
- ``∂lvar/∂θ``, ``∂uvar/∂θ``, ``∂lcon/∂θ``, ``∂ucon/∂θ``: the bounds with respect to the parameters, `nvar × npar` and `ncon × npar`, see [`lvar_jac_par`](@ref)

Sparse coordinate format follows `NLPModels`: `(rows, cols, vals)` with rows in the constraint or variable index and
columns in the parameter index, repeated entries summed. Dimensions and availability come from
[`ParametricNLPModelMeta`](@ref) through [`get_par_meta`](@ref).
"""
module ParametricNLPModels

using NLPModels
using SparseArrays

const OBJECTIVE_HESSIAN_PAR = raw"""
```math
σ ∂²f/∂x∂θ(x),
```
"""

const LAGRANGIAN_HESSIAN_PAR = raw"""
```math
∂²L/∂x∂θ(x,y) = σ ∂²f/∂x∂θ(x) + \sum_i yᵢ ∂²cᵢ/∂x∂θ(x),
```
"""

include("meta.jl")
include("api.jl")
include("bounds.jl")

end
