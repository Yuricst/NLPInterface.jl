# The SNOPT ccall plumbing against a mock library (test/mock_snopt), so that
# it runs in CI without a licensed SNOPT. Needs a C compiler; CI sets
# NLPINTERFACE_REQUIRE_MOCK_SNOPT=1 where one must be available. On Windows
# (DLL build and loading untested) it runs only when that variable is set.
@testset "mock SNOPT library" begin
    require_mock = get(ENV, "NLPINTERFACE_REQUIRE_MOCK_SNOPT", "") == "1"
    include(joinpath(@__DIR__, "mock_snopt", "build.jl"))
    mktempdir() do root
        libs = Sys.iswindows() && !require_mock ? nothing : build_mock_snopt(root)
        if libs === nothing
            if require_mock
                error("NLPINTERFACE_REQUIRE_MOCK_SNOPT=1 but the mock SNOPT library could not be built")
            end
            @test_skip "mock SNOPT library not built"
        else
            script = joinpath(@__DIR__, "mock_snopt", "runtests_mock.jl")
            for variant in (:sninitx, :fallback)
                cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $variant`
                cmd = addenv(cmd, "SNOPT_LIB" => libs[variant])
                @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
            end
        end
    end
end
