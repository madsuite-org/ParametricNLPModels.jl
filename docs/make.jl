using Documenter
using ParametricNLPModels

makedocs(
    sitename = "ParametricNLPModels.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX(),
        size_threshold_ignore = ["reference.md"],
    ),
    modules = [ParametricNLPModels],
    repo = Documenter.Remotes.GitHub("madsuite-org", "ParametricNLPModels.jl"),
    checkdocs = :exports,
    clean = true,
    pages = [
        "Home" => "index.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(
    repo = "github.com/madsuite-org/ParametricNLPModels.jl.git",
    devbranch = "main",
    push_preview = true,
)
