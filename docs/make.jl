"""
Make documentation with Documenter.jl
"""

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter

include(joinpath(dirname(@__FILE__), "../src/NLPInterface.jl"))

makedocs(
    clean = true,
    build = joinpath(@__DIR__, "build"),
    modules = [NLPInterface],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main",
        canonical = "https://yuricst.github.io/NLPInterface.jl",
    ),
    sitename = "NLPInterface.jl",
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API" => "api.md",
    ],
)
