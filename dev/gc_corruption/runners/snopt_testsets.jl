# Step 4: run test_snopt.jl with only some of its @testsets, then force full
# GCs (with allocation in between) so that damage is detected before exit.
#   NLPI_TESTDIR       test directory
#   NLPI_KEEP          comma-separated @testset names to keep (default: all)
#   NLPI_PRE_FILES     test files to include before (comma-separated, optional)
#   NLPI_REPEAT        how many times to run the kept testsets (default 1)
using NLPInterface
using ForwardDiff
using Test

const TESTDIR = ENV["NLPI_TESTDIR"]
const KEEP = haskey(ENV, "NLPI_KEEP") ? Set(split(ENV["NLPI_KEEP"], ",")) : nothing

testset_name(ex) = (ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@testset")) ?
    something(findfirst(a -> a isa String, ex.args[2:end]) |> i -> i === nothing ? nothing : ex.args[1 + i], "") : nothing

# Drop the @testsets whose name is not kept, at top level and inside `if` blocks.
function filter_testsets(ex)
    ex isa Expr || return ex
    name = testset_name(ex)
    if name !== nothing
        return (KEEP === nothing || name in KEEP) ? ex : nothing
    end
    ex.head in (:toplevel, :block, :if) || return ex
    return Expr(ex.head, map(filter_testsets, ex.args)...)
end

src = read(joinpath(TESTDIR, "test_snopt.jl"), String)
code = filter_testsets(Meta.parseall(src; filename=joinpath(TESTDIR, "test_snopt.jl")))
println("snopt_testsets.jl: julia ", VERSION, " threads ", Threads.nthreads(), "+",
    Threads.nthreads(:interactive), " keep=", KEEP === nothing ? "all" : join(sort!(collect(KEEP)), ","))
flush(stdout)

cd(TESTDIR) do
    include(joinpath(TESTDIR, "problems.jl"))
    include(joinpath(TESTDIR, "helpers.jl"))
    for f in filter(!isempty, split(get(ENV, "NLPI_PRE_FILES", ""), ","))
        include(joinpath(TESTDIR, f))
    end
    for _ in 1:parse(Int, get(ENV, "NLPI_REPEAT", "1"))
        Core.eval(Main, code)
    end
end
# Detection: full collections with fresh allocation in between.
for i in 1:20
    v = [Vector{Any}(undef, 100) for _ in 1:10_000]
    GC.gc(true)
end
println("snopt_testsets.jl: completed")
