using Test
using NLPModels
using NLPModelsParametric
using SparseArrays

# f(x) = θ₁ x₁² + x₂ and c(x) = θ₂ x₁ x₂
struct TestModel{T} <: AbstractNLPModel{T, Vector{T}}
    meta::NLPModelMeta{T, Vector{T}}
    counters::Counters
    θ::Vector{T}
end
TestModel(θ::Vector) = TestModel(NLPModelMeta(2; ncon = 1, nnzj = 2, nnzh = 1), Counters(), θ)

NLPModelsParametric.get_npar(nlp::TestModel) = 2
NLPModelsParametric.get_nnzj_par(nlp::TestModel) = 1
NLPModelsParametric.get_nnzh_par(nlp::TestModel) = 3
NLPModelsParametric.get_par(nlp::TestModel) = nlp.θ
NLPModelsParametric.set_par!(nlp::TestModel, θ) = (copyto!(nlp.θ, θ); nlp)
function NLPModelsParametric.jac_par_structure!(nlp::TestModel, rows, cols)
    rows[1], cols[1] = 1, 2
    return rows, cols
end
function NLPModelsParametric.jac_par_coord!(nlp::TestModel, x, vals)
    vals[1] = x[1] * x[2]
    return vals
end
function NLPModelsParametric.hess_par_structure!(nlp::TestModel, rows, cols)
    copyto!(rows, [1, 1, 2])
    copyto!(cols, [1, 2, 2])
    return rows, cols
end
function NLPModelsParametric.hess_par_coord!(nlp::TestModel, x, y, vals; obj_weight = one(eltype(x)))
    copyto!(vals, [2obj_weight * x[1], y[1] * x[2], y[1] * x[1]])
    return vals
end

struct PlainModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
end

@testset "NLPModelsParametric" begin
    nlp = TestModel([2.0, 3.0])
    x, y, w, v = [1.5, -2.0], [0.5], 0.3, [1.0, 2.0]
    J = [0.0 -3.0]
    H = [3.0 -1.0; 0.0 0.75]
    Hw = [2w * x[1] -1.0; 0.0 0.75]

    @test jac_par_structure(nlp) == ([1], [2])
    @test jac_par_coord(nlp, x) == [-3.0]
    @test jac_par(nlp, x) isa SparseMatrixCSC
    @test jac_par(nlp, x) == J
    @test jac_par_dense!(nlp, x, ones(1, 2)) == J
    @test jprod_par(nlp, x, v) == [-6.0]
    @test jprod_par!(nlp, [1], [2], [-3.0], v, zeros(1)) == [-6.0]
    @test jtprod_par(nlp, x, [2.0]) == [0.0, -6.0]
    @test jtprod_par!(nlp, [1], [2], [-3.0], [2.0], zeros(2)) == [0.0, -6.0]

    @test hess_par_structure(nlp) == ([1, 1, 2], [1, 2, 2])
    @test hess_par_coord(nlp, x, y) == [3.0, -1.0, 0.75]
    @test hess_par(nlp, x, y) isa SparseMatrixCSC
    @test hess_par(nlp, x, y) == H
    @test hess_par(nlp, x, y; obj_weight = w) == Hw
    @test hess_par_dense!(nlp, x, y, ones(2, 2)) == H
    @test hess_par_dense!(nlp, x, y, ones(2, 2); obj_weight = w) == Hw
    @test hprod_par(nlp, x, y, v) == [1.0, 1.5]
    @test hprod_par(nlp, x, y, v; obj_weight = w) == [-1.1, 1.5]
    @test hprod_par!(nlp, [1, 1, 2], [1, 2, 2], [3.0, -1.0, 0.75], v, zeros(2)) == [1.0, 1.5]
    @test htprod_par(nlp, x, y, v) == [3.0, 0.5]
    @test htprod_par!(nlp, [1, 1, 2], [1, 2, 2], [3.0, -1.0, 0.75], v, zeros(2)) == [3.0, 0.5]

    @test get_par(nlp) == [2.0, 3.0]
    set_par!(nlp, [4.0, 5.0])
    @test get_par(nlp) == [4.0, 5.0]

    @test NLPModelsParametric._dense!(zeros(1, 2), [1, 1], [2, 2], [1.0, 2.0]) == [0.0 3.0]

    @test_throws DimensionError jac_par_coord(nlp, [1.0])
    @test_throws DimensionError hess_par_coord(nlp, x, [1.0, 2.0])
    @test_throws DimensionError jprod_par(nlp, x, [1.0])
    @test_throws DimensionError htprod_par(nlp, x, y, [1.0])
    @test_throws MethodError jac_par(PlainModel(NLPModelMeta(2), Counters()), x)
end
