# Run a subset of the test files, in order, like test/runtests.jl.
#   NLPI_TESTDIR  test directory (default: test/ of this repository)
#   NLPI_FILES    comma-separated test files (default: all of runtests.jl)
using NLPInterface
using ForwardDiff
using Test

const TESTDIR = get(ENV, "NLPI_TESTDIR", joinpath(@__DIR__, "..", "..", "..", "test"))
const ALL = ["test_problems.jl", "test_validation.jl", "test_residuals.jl", "test_sparse.jl",
    "test_ipopt.jl", "test_snopt.jl", "test_mock_snopt.jl", "test_review_fixes.jl"]
files = haskey(ENV, "NLPI_FILES") ? split(ENV["NLPI_FILES"], ",") : ALL
files = filter(f -> isfile(joinpath(TESTDIR, f)), files)
println("subset.jl: julia ", VERSION, ", threads ", Threads.nthreads(), "+", Threads.nthreads(:interactive),
    ", has_snopt=", has_snopt(), ", files: ", join(files, " "))

cd(TESTDIR) do
    include(joinpath(TESTDIR, "problems.jl"))
    include(joinpath(TESTDIR, "helpers.jl"))
    isfile(joinpath(TESTDIR, "transcription.jl")) && include(joinpath(TESTDIR, "transcription.jl"))
    @testset "NLPInterface" begin
        for f in files
            include(joinpath(TESTDIR, f))
        end
    end
end
