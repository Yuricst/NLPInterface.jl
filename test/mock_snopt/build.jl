# Build the mock SNOPT libraries (mock_core.c, mock_wrapper.c) with the system
# C compiler. Used by test/test_mock_snopt.jl; see test/mock_snopt/runtests_mock.jl.

using Libdl

function find_c_compiler()
    for name in (get(ENV, "CC", ""), "cc", "gcc", "clang")
        isempty(name) && continue
        path = Sys.which(name)
        path === nothing || return path
    end
    return nothing
end

"""
    build_mock_snopt(root) -> Union{Nothing,Dict{Symbol,String}}

Compile two mock SNOPT installations under `root` and return the path of each
wrapper library:

- `:sninitx`: `libsnopt7_cpp` exporting `f_sninitx` (the normal path);
- `:fallback`: `libsnopt7_c` without `f_sninitx`, so that NLPInterface uses
  `sninit_` and the file helpers of the mock core.

Each directory also holds the mock core `libsnopt7`, which the wrapper links
against. On macOS the core's install name points to a missing directory, as in
prebuilt SNOPT distributions, so that loading exercises
`relocate_snopt_library`. Returns `nothing` (after a warning) if there is no C
compiler or the build fails.
"""
function build_mock_snopt(root::AbstractString)
    cc = find_c_compiler()
    if cc === nothing
        @warn "No C compiler found; cannot build the mock SNOPT library"
        return nothing
    end
    src = @__DIR__
    ext = Libdl.dlext
    flags = ["-std=c99", "-O1", "-shared"]
    Sys.iswindows() || push!(flags, "-fPIC")
    libs = Dict{Symbol,String}()
    try
        for (variant, wrapname, defs) in (
            (:sninitx, "libsnopt7_cpp", ["-DMOCK_HAS_SNINITX"]),
            (:fallback, "libsnopt7_c", String[]),
        )
            dir = joinpath(root, String(variant))
            mkpath(dir)
            core = joinpath(dir, "libsnopt7.$ext")
            wrap = joinpath(dir, "$wrapname.$ext")
            core_flags = if Sys.isapple()
                ["-install_name", "/nonexistent/NLPInterface-mock/$(variant)/libsnopt7.$ext"]
            elseif Sys.islinux()
                ["-Wl,-soname,libsnopt7.$ext"]
            else
                String[]
            end
            run(`$cc $flags $core_flags -o $core $(joinpath(src, "mock_core.c"))`)
            wrap_flags = if Sys.islinux()
                # RPATH (not RUNPATH, which LD_LIBRARY_PATH overrides), so a real
                # SNOPT on LD_LIBRARY_PATH cannot stand in for the mock core.
                ["-Wl,-rpath,\$ORIGIN", "-Wl,--disable-new-dtags"]
            elseif Sys.isapple()
                ["-headerpad_max_install_names"]
            else
                String[]
            end
            run(`$cc $flags $defs $wrap_flags -o $wrap $(joinpath(src, "mock_wrapper.c")) -L$dir -lsnopt7 -lm`)
            libs[variant] = wrap
        end
    catch err
        @warn "Building the mock SNOPT library failed" exception = err
        return nothing
    end
    return libs
end
