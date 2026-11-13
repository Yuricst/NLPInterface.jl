using NLPInterface
using ForwardDiff
using Test

include("problems.jl")
include("helpers.jl")
include("transcription.jl")

@testset "NLPInterface" begin
    include("test_problems.jl")      # end-to-end solves of problems.jl
    include("test_validation.jl")    # src/validation.jl
    include("test_residuals.jl")     # src/residuals.jl
    include("test_sparse.jl")        # src/sparsity.jl
    include("test_ipopt.jl")         # src/ipopt.jl
    include("test_snopt.jl")         # src/snopt.jl
    include("test_mock_snopt.jl")    # src/snopt.jl against test/mock_snopt
    include("test_review_fixes.jl")

    if !has_snopt()
        @info "Skipping SNOPT tests; set SNOPT_LIB or SNOPTDIR to the libsnopt7_cpp library"
        @test_skip "SNOPT library not found"
    end
end
