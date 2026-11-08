"""Interface for SNOPT7"""

# SNOPT inform codes: SNOPT 7.7 User's Guide, Section 8.6
# (https://ccom.ucsd.edu/~optimizers/docs/snopt/).
# Codes 1-3 return a feasible point (2: only with the `Feasible point` option;
# 3: feasible, but within 1e-2 of the requested optimality tolerance).
# The manual lists 5 and 6 as normal endings of elastic mode, where elastic
# constraints may be violated, and does not state the constraints hold, so they
# get their own symbols instead of `:Solved_To_Acceptable_Level`. Code 4 (weak QP
# minimizer) is not in the snOptA list and is not expected; it is mapped for
# completeness.
const SNOPT_RETURN_STATUS = Dict(
    1 => :Solve_Succeeded,
    2 => :Feasible_Point_Found,
    3 => :Solved_To_Acceptable_Level,
    4 => :Weak_QP_Minimizer,
    5 => :Elastic_Objective_Minimized,
    6 => :Elastic_Infeasibilities_Minimized,
    11 => :Infeasible_Problem_Detected,
    12 => :Infeasible_Problem_Detected,
    13 => :Infeasible_Problem_Detected,
    14 => :Infeasible_Problem_Detected,
    15 => :Infeasible_Problem_Detected,
    16 => :Infeasible_Problem_Detected,
    21 => :Unbounded_Problem_Detected,
    22 => :Unbounded_Problem_Detected,
    31 => :Maximum_Iterations_Exceeded,
    32 => :Maximum_Iterations_Exceeded,
    33 => :Superbasics_Limit_Too_Small,
    34 => :Maximum_CpuTime_Exceeded,
    41 => :Numerical_Difficulties,
    42 => :Numerical_Difficulties,
    43 => :Numerical_Difficulties,
    44 => :Numerical_Difficulties,
    45 => :Numerical_Difficulties,
    51 => :User_Supplied_Function_Error,
    52 => :User_Supplied_Function_Error,
    56 => :User_Supplied_Function_Error,
    61 => :User_Supplied_Function_Undefined,
    62 => :User_Supplied_Function_Undefined,
    63 => :User_Supplied_Function_Undefined,
    71 => :User_Requested_Stop,
    72 => :User_Requested_Stop,
    73 => :User_Requested_Stop,
    74 => :User_Requested_Stop,
    81 => :Insufficient_Memory,
    82 => :Insufficient_Memory,
    83 => :Insufficient_Memory,
    84 => :Insufficient_Memory,
    91 => :Invalid_Problem_Definition,
    92 => :Invalid_Problem_Definition,
    141 => :Internal_Error,
    142 => :Internal_Error,
    999 => :Internal_Error,
)

const SNOPT_INF = 1.0e20
const SNOPT_REQUIRED_SYMBOLS = (
    :f_snend,
    :f_snset,
    :f_snseti,
    :f_snsetr,
    :f_snmema,
    :f_snopta,
)
const SNOPT_LOCK = ReentrantLock()
# A solve that waits this long for `SNOPT_LOCK` warns about a possible deadlock.
const SNOPT_LOCK_WAIT_WARN_SECONDS = Ref(60.0)
const SNOPT_TEMPDIRS = String[]
const SNOPT_OPEN_HANDLES = Ptr{Cvoid}[]
# Fortran units used for the Print and Summary files on the `sninit_` fallback
# path (libraries without `f_sninitx`). Unit 6 is standard output.
const SNOPT_FALLBACK_PRINT_UNIT = Cint(9)
const SNOPT_FALLBACK_SUMMARY_UNIT = Cint(10)
const SNOPT_CONSOLE_UNIT = Cint(6)
# `f_snmema` INFO when the workspace estimate succeeded (snMemA, SNOPT 7.7).
const SNOPT_MEMA_OK = 104

# Resolved at runtime in `__init__`; ccall re-reads this global on each call.
libsnopt7 = ""
libsnopt7_core = ""
libsnopt7_has_sninitx = false

# The user callback of the running solve. SNOPT's C interface keeps global state
# (for example the character workspace `__snopt_wrapper_MOD_cw` and the Fortran
# I/O units of the Print and Summary files), so only one solve may run at a time
# (`SNOPT_LOCK`) and nested solves are rejected (`_snopt_in_solve`).
const _SNOPT_USRFUN = Ref{Any}(nothing)
const _SNOPT_CALLBACK_ERROR = Ref{Any}(nothing)
const _SNOPT_CALLBACK_BACKTRACE = Ref{Any}(nothing)
# Last `DomainError` treated as "undefined here" (Status = -1), for the warning
# issued when SNOPT cannot recover from it.
const _SNOPT_LAST_DOMAIN_ERROR = Ref{Any}(nothing)

# Marks code running inside a `solve_snopt`: the solve itself, its `fitness`
# callbacks and, where ScopedValues exist (Julia >= 1.11), tasks spawned from
# them. A nested solve from such a task would deadlock on `SNOPT_LOCK` (other
# task) or overwrite the outer solve's state (same task), so it throws instead.
# The scoped value is a per-solve flag that is cleared when the solve ends, so a
# task spawned from `fitness` that outlives the solve may call `solve_snopt`.
@static if isdefined(Base, :ScopedValues)
    const _SNOPT_IN_SOLVE = Base.ScopedValues.ScopedValue{Union{Nothing,Base.RefValue{Bool}}}(nothing)
    function _snopt_in_solve()
        active = _SNOPT_IN_SOLVE[]
        return active !== nothing && active[]
    end
    function _with_snopt_in_solve(f)
        active = Ref(true)
        try
            return Base.ScopedValues.with(f, _SNOPT_IN_SOLVE => active)
        finally
            active[] = false
        end
    end
else
    # Julia 1.10: task-local only, so tasks spawned from `fitness` are not covered.
    _snopt_in_solve() = get(task_local_storage(), :NLPInterface_snopt_in_solve, false)::Bool
    _with_snopt_in_solve(f) = task_local_storage(f, :NLPInterface_snopt_in_solve, true)
end

function check_snopt_not_nested()
    _snopt_in_solve() || return nothing
    throw(
        ErrorException(
            "solve_snopt was called from inside the fitness function of another " *
            "solve_snopt (or from a task spawned there). SNOPT's C interface keeps " *
            "global state, so nested solves are not supported; use solve_ipopt for " *
            "the inner problem, or restructure the problem.",
        ),
    )
end

# What `usrfun` should report to SNOPT when `fitness` throws `err`.
# Status = -1 means "the functions are undefined at x": SNOPT shortens the step
# and continues, so the error is not rethrown (if SNOPT cannot recover it ends
# with INFO 61-63, `:User_Supplied_Function_Undefined`). Only a `DomainError`
# (sqrt or log of a negative number, ...) is treated this way. Every other
# error, including `InterruptException` (Ctrl-C), uses Status = -2, which stops
# SNOPT immediately; `solve_snopt` then rethrows the error. A `DomainError`
# thrown in a task spawned by `fitness` arrives wrapped in a
# `TaskFailedException` (or a `CompositeException` from `@sync`) and counts too.
snopt_callback_status(err) = is_domain_error(err) ? Cint(-1) : Cint(-2)

is_domain_error(::DomainError) = true
is_domain_error(err::TaskFailedException) = is_domain_error(err.task.exception)
is_domain_error(err::CapturedException) = is_domain_error(err.ex)
function is_domain_error(err::CompositeException)
    return !isempty(err.exceptions) && all(is_domain_error, err.exceptions)
end
is_domain_error(_) = false

function snopt_bound(x)
    v = Float64(x)
    return isfinite(v) ? v : copysign(SNOPT_INF, v)
end

function snopt_libnames()
    ext = Libdl.dlext
    return (
        string("libsnopt7_cpp.", ext),
        string("libsnopt7_c.", ext),
        string("libsnopt7.", ext),
    )
end

function preload_snopt_companions(libpath::AbstractString)
    dir = dirname(abspath(libpath))
    for name in ("libsnopt7." * Libdl.dlext, "libgfortran.5." * Libdl.dlext)
        companion = joinpath(dir, name)
        isfile(companion) || continue
        handle = Libdl.dlopen_e(companion)
        if handle != C_NULL
            push!(SNOPT_OPEN_HANDLES, handle)
        end
    end
    return nothing
end

function snopt_missing_install_names(libpath::AbstractString)
    Sys.isapple() || return String[]
    otool = Sys.which("otool")
    otool === nothing && return String[]
    deps = String[]
    for line in eachline(`$otool -L $libpath`)
        m = match(r"^\s+(\S+)", line)
        m === nothing && continue
        dep = m.captures[1]
        occursin("libsnopt7", basename(dep)) || continue
        basename(dep) == basename(libpath) && continue
        isfile(dep) || push!(deps, dep)
    end
    return unique(deps)
end

# Directory for the relocated copies of `libpath` and its `libsnopt7` core,
# keyed on both files' paths, sizes and modification times, under this
# package's scratch space in the first depot. `nothing` if there is no depot.
function snopt_relocation_cache_dir(libpath::AbstractString, core::AbstractString)
    isempty(DEPOT_PATH) && return nothing
    key = try
        hash((
            realpath(libpath), filesize(libpath), mtime(libpath),
            realpath(core), filesize(core), mtime(core),
        ))
    catch
        return nothing
    end
    return joinpath(
        first(DEPOT_PATH), "scratchspaces", "e604ab98-cb68-4d52-a41a-b2fbd5dba595",
        "snopt_relocated", string(key; base=16),
    )
end

# Copy the wrapper and its core into `dir`, point the wrapper's missing
# `libsnopt7` install names at the copy next to it, and re-sign both. Returns
# the path of the copied wrapper, or `nothing` if a tool failed.
function build_relocated_snopt(dir, libpath, core, missing_deps, tool)
    tmpcore = joinpath(dir, "libsnopt7." * Libdl.dlext)
    tmpwrap = joinpath(dir, basename(libpath))
    try
        cp(core, tmpcore; follow_symlinks=true, force=true)
        cp(libpath, tmpwrap; follow_symlinks=true, force=true)
    catch
        return nothing  # disk full, quota, ...: never fail the package load
    end
    for old in missing_deps
        try
            run(pipeline(`$tool -change $old @loader_path/libsnopt7.$(Libdl.dlext) $tmpwrap`; stderr=devnull))
        catch
            return nothing
        end
    end
    codesign = Sys.which("codesign")
    if codesign !== nothing
        try
            run(pipeline(`$codesign --force --sign - $tmpwrap`; stdout=devnull, stderr=devnull))
            run(pipeline(`$codesign --force --sign - $tmpcore`; stdout=devnull, stderr=devnull))
        catch
            return nothing
        end
    end
    return tmpwrap
end

# On macOS, a wrapper whose `libsnopt7` install name points to a missing file
# (common in prebuilt SNOPT distributions) cannot be loaded as is. Load a copy
# whose install name points to a copy of the core next to it. The copies are
# made once and cached in the package's scratch space, so later sessions load
# them directly; a temporary directory is used if the cache cannot be written.
function relocate_snopt_library(libpath::AbstractString)
    Sys.isapple() || return String(libpath)
    missing_deps = snopt_missing_install_names(libpath)
    isempty(missing_deps) && return String(libpath)
    dir = dirname(abspath(libpath))
    core = joinpath(dir, "libsnopt7." * Libdl.dlext)
    isfile(core) || return String(libpath)
    tool = Sys.which("install_name_tool")
    tool === nothing && return String(libpath)

    cache = snopt_relocation_cache_dir(libpath, core)
    if cache !== nothing
        cached = joinpath(cache, basename(libpath))
        isfile(cached) && isfile(joinpath(cache, "libsnopt7." * Libdl.dlext)) && return cached
        # An incomplete cache entry (e.g. a file deleted by hand) would block
        # the rename below forever; move it aside and rebuild. Recheck first,
        # so that an entry another session has just completed is kept.
        complete() = isfile(cached) && isfile(joinpath(cache, "libsnopt7." * Libdl.dlext))
        if isdir(cache) && !complete()
            try
                stale = cache * ".stale-" * string(hash((time_ns(), getpid())); base=16)
                mv(cache, stale)
                rm(stale; recursive=true, force=true)
            catch
            end
        end
        complete() && return cached
        # Build in a private directory and rename it into place, so concurrent
        # Julia sessions never load a half-written copy.
        staged = try
            mkpath(dirname(cache))
            mktempdir(dirname(cache); cleanup=false)
        catch
            nothing
        end
        if staged !== nothing
            built = build_relocated_snopt(staged, libpath, core, missing_deps, tool)
            if built !== nothing
                try
                    mv(staged, cache)
                catch
                    # Another session finished first; use its copy.
                end
            end
            rm(staged; recursive=true, force=true)
            isfile(cached) && return cached
        end
    end

    tmp = try
        mktempdir()
    catch
        return String(libpath)
    end
    push!(SNOPT_TEMPDIRS, tmp)
    built = build_relocated_snopt(tmp, libpath, core, missing_deps, tool)
    return built === nothing ? String(libpath) : built
end

function library_has_snopt_c_api(handle::Ptr{Cvoid})
    for sym in SNOPT_REQUIRED_SYMBOLS
        Libdl.dlsym_e(handle, sym) == C_NULL && return false
    end
    has_initx = Libdl.dlsym_e(handle, :f_sninitx) != C_NULL
    has_init = Libdl.dlsym_e(handle, :f_sninit) != C_NULL
    return has_initx || has_init
end

function loadable_snopt_library(libpath::AbstractString)
    isempty(libpath) && return "", false
    prepared = String(libpath)
    if isfile(libpath)
        # Never let the macOS relocation (external tools, file copies) break
        # the package load; fall back to loading the library as is.
        prepared = try
            relocate_snopt_library(libpath)
        catch
            String(libpath)
        end
        preload_snopt_companions(prepared)
    end
    handle = Libdl.dlopen_e(prepared)
    handle == C_NULL && return "", false
    if !library_has_snopt_c_api(handle)
        Libdl.dlclose(handle)
        return "", false
    end
    has_initx = Libdl.dlsym_e(handle, :f_sninitx) != C_NULL
    push!(SNOPT_OPEN_HANDLES, handle)
    return String(Libdl.dlpath(handle)), has_initx
end

function snopt_search_dirs()
    dirs = String[]
    if haskey(ENV, "SNOPT_LIB") && !isempty(ENV["SNOPT_LIB"])
        push!(dirs, dirname(abspath(ENV["SNOPT_LIB"])))
    end
    if haskey(ENV, "SNOPTDIR") && !isempty(ENV["SNOPTDIR"])
        push!(dirs, ENV["SNOPTDIR"])
    end
    pkgroot = dirname(@__DIR__)
    push!(dirs, joinpath(pkgroot, "deps", "usr", "lib"))
    # Empty entries in a search path (`a::b`, a trailing `:`) would turn into
    # bare library names; the system loader is tried separately at the end.
    pathlist(var, sep) = filter!(!isempty, split(get(ENV, var, ""), sep))
    if Sys.iswindows()
        append!(dirs, pathlist("PATH", ';'))
    else
        append!(dirs, pathlist("LD_LIBRARY_PATH", ':'))
        Sys.isapple() && append!(dirs, pathlist("DYLD_LIBRARY_PATH", ':'))
    end
    return dirs
end

function remember_snopt_core!(path::AbstractString)
    core = joinpath(dirname(String(path)), "libsnopt7." * Libdl.dlext)
    global libsnopt7_core = isfile(core) ? core : String(path)
    return nothing
end

"""
    find_snopt_lib() -> String

Search for a loadable SNOPT shared library containing the C `f_*` wrappers.
Prefers `libsnopt7_cpp`, then `libsnopt7_c` / `libsnopt7`. Returns `""` if none
is found. Sets `libsnopt7_has_sninitx` when the chosen library exports `f_sninitx`.
"""
function find_snopt_lib()
    candidates = String[]
    if haskey(ENV, "SNOPT_LIB") && !isempty(ENV["SNOPT_LIB"])
        push!(candidates, ENV["SNOPT_LIB"])
    end
    for dir in snopt_search_dirs()
        for name in snopt_libnames()
            push!(candidates, joinpath(dir, name))
        end
    end
    seen_unusable = String[]
    for cand in unique(candidates)
        path, has_initx = loadable_snopt_library(cand)
        if !isempty(path)
            global libsnopt7_has_sninitx = has_initx
            remember_snopt_core!(path)
            return path
        end
        isfile(cand) && push!(seen_unusable, cand)
    end
    for name in ("libsnopt7_cpp", "libsnopt7_c", "libsnopt7")
        path, has_initx = loadable_snopt_library(name)
        if !isempty(path)
            global libsnopt7_has_sninitx = has_initx
            remember_snopt_core!(path)
            return path
        end
    end
    if !isempty(seen_unusable)
        @warn "SNOPT library files were found but could not be loaded" candidates = seen_unusable julia_arch = Sys.ARCH
    end
    global libsnopt7_has_sninitx = false
    global libsnopt7_core = ""
    return ""
end

# `true` if the `sninit_` fallback can open Print and Summary files itself, with
# the Fortran helpers in the core `libsnopt7` library.
function snopt_fallback_opens_files()
    isempty(libsnopt7_core) && return false
    handle = Libdl.dlopen_e(libsnopt7_core)
    handle == C_NULL && return false
    try
        return Libdl.dlsym_e(handle, :snfileopenwrite_) != C_NULL &&
               Libdl.dlsym_e(handle, :snfileclose_) != C_NULL
    finally
        Libdl.dlclose(handle)
    end
end

# Fortran unit for one output file on the `sninit_` fallback path: 0 for none,
# 6 for the console, otherwise `unit` after opening the file on it (recorded in
# `opened` so that it is closed after the solve).
function snopt_fallback_unit!(opened, path, unit)
    name = _snopt_trim(path)
    isempty(name) && return Cint(0)
    _snopt_is_console_name(name) && return SNOPT_CONSOLE_UNIT
    inform = Ref{Cint}(-1)
    ccall(
        (:snfileopenwrite_, libsnopt7_core),
        Cvoid,
        (Ref{Cint}, Ptr{UInt8}, Ref{Cint}, Csize_t),
        unit,
        name,
        inform,
        ncodeunits(name),
    )
    inform[] == 0 || throw(ArgumentError("SNOPT could not open $(repr(name)) (INFO $(inform[]))"))
    push!(opened, unit)
    return unit
end

function snopt_close_fallback_units!(opened)
    for unit in opened
        ccall((:snfileclose_, libsnopt7_core), Cvoid, (Ref{Cint},), unit)
    end
    empty!(opened)
    return nothing
end

function snopt_init_workspace!(printpath, summpath, cw, iw, rw, opened)
    if libsnopt7_has_sninitx
        ccall(
            (:f_sninitx, libsnopt7),
            Cvoid,
            (Cstring, Cint, Cstring, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
            printpath,
            ncodeunits(printpath),
            summpath,
            ncodeunits(summpath),
            iw,
            length(iw),
            rw,
            length(rw),
        )
    else
        # Some C wrappers export `f_sninit` with a broken ABI. The Fortran
        # `sninit_` in the companion libsnopt7 library is reliable. `solve_snopt`
        # has cleared the file names already if they cannot be opened here.
        if isempty(libsnopt7_core)
            global libsnopt7_core = libsnopt7
        end
        iprint = Ref{Cint}(snopt_fallback_unit!(opened, printpath, SNOPT_FALLBACK_PRINT_UNIT))
        isumm = Ref{Cint}(snopt_fallback_unit!(opened, summpath, SNOPT_FALLBACK_SUMMARY_UNIT))
        lencw = Ref{Cint}(div(length(cw), 8))
        leniw = Ref{Cint}(length(iw))
        lenrw = Ref{Cint}(length(rw))
        ccall(
            (:sninit_, libsnopt7_core),
            Cvoid,
            (Ref{Cint}, Ref{Cint}, Ptr{UInt8}, Ref{Cint}, Ptr{Cint}, Ref{Cint}, Ptr{Cdouble}, Ref{Cint}),
            iprint,
            isumm,
            cw,
            lencw,
            iw,
            leniw,
            rw,
            lenrw,
        )
    end
    return nothing
end

"""
    has_snopt() -> Bool

Return `true` if a usable SNOPT shared library was loaded at startup.
"""
has_snopt() = !isempty(libsnopt7)

function require_snopt()
    has_snopt() && return libsnopt7
    throw(
        ErrorException(
            "SNOPT library not found. Set SNOPT_LIB to the full path of " *
            "libsnopt7_cpp, or SNOPTDIR to the directory containing it.",
        ),
    )
end

function snopt_option_keyword(key::Symbol)
    return replace(String(key), "_" => " ")
end
function snopt_option_keyword(key::AbstractString)
    return String(key)
end

function snopt_set_option!(iw, leniw, rw, lenrw, optstring::AbstractString)
    errors = Ref{Cint}(0)
    ccall(
        (:f_snset, libsnopt7),
        Cvoid,
        (Cstring, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
        optstring,
        ncodeunits(optstring),
        errors,
        iw,
        leniw,
        rw,
        lenrw,
    )
    errors[] == 0 || throw(ArgumentError("SNOPT rejected option $(repr(optstring))"))
    return nothing
end

function snopt_set_option!(iw, leniw, rw, lenrw, keyword::AbstractString, value::Integer)
    errors = Ref{Cint}(0)
    ccall(
        (:f_snseti, libsnopt7),
        Cvoid,
        (Cstring, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
        keyword,
        ncodeunits(keyword),
        Int32(value),
        errors,
        iw,
        leniw,
        rw,
        lenrw,
    )
    errors[] == 0 || throw(ArgumentError("SNOPT rejected option $(repr(keyword)) => $value"))
    return nothing
end

function snopt_set_option!(iw, leniw, rw, lenrw, keyword::AbstractString, value::AbstractFloat)
    errors = Ref{Cint}(0)
    ccall(
        (:f_snsetr, libsnopt7),
        Cvoid,
        (Cstring, Cint, Cdouble, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
        keyword,
        ncodeunits(keyword),
        Float64(value),
        errors,
        iw,
        leniw,
        rw,
        lenrw,
    )
    errors[] == 0 || throw(ArgumentError("SNOPT rejected option $(repr(keyword)) => $value"))
    return nothing
end

function apply_snopt_option!(iw, leniw, rw, lenrw, key, val)
    keyword = snopt_option_keyword(key)
    if val isa Bool
        throw(ArgumentError("SNOPT option $(repr(keyword)) does not accept Bool values"))
    elseif val isa Integer
        snopt_set_option!(iw, leniw, rw, lenrw, keyword, val)
    elseif val isa AbstractFloat
        snopt_set_option!(iw, leniw, rw, lenrw, keyword, val)
    elseif val isa AbstractString
        snopt_set_option!(iw, leniw, rw, lenrw, string(keyword, " ", val))
    elseif val isa Symbol
        snopt_set_option!(iw, leniw, rw, lenrw, string(keyword, " ", snopt_option_keyword(val)))
    else
        throw(ArgumentError("Unsupported SNOPT option type $(typeof(val)) for $(repr(keyword))"))
    end
    return nothing
end

function _snopt_console_summary(major_print_level, summfile::AbstractString)
    isempty(summfile) || return _snopt_is_console_name(summfile)
    major_print_level === nothing && return false
    if major_print_level isa Number
        return major_print_level != 0
    end
    return true
end

function snopt_output_files(
    printfile::AbstractString,
    summfile::AbstractString;
    console_summary::Bool=false,
)
    # An empty name means "no file": f_sninitx then leaves that unit closed.
    # Do not substitute /dev/null here. Opening the same path on both the Print
    # and Summary units makes the Fortran OPEN fail, f_snopta then returns
    # INFO 91 without evaluating anything, and the failed open can leave later
    # solves in the same process failing too.
    printpath = String(printfile)
    if !isempty(summfile)
        summpath = String(summfile)
    elseif console_summary
        # snInitF / f_sninitx: "screen" routes the Summary file to stdout.
        summpath = "screen"
    else
        summpath = ""
    end
    return printpath, summpath
end

# SNOPT (Fortran) trims trailing blanks from file names but keeps leading ones,
# so all comparisons below use `rstrip`, never `strip`.
_snopt_trim(path::AbstractString) = rstrip(String(path))

# Only the exact word "screen" (after trimming trailing blanks) means the
# console. f_sninitx writes to a file with any other name, including "stdout"
# and "-"; the `sninit_` fallback path follows the same rule.
_snopt_is_console_name(path::AbstractString) = _snopt_trim(path) == "screen"

# Normalized key for comparing output file names: absolute path with the
# directory resolved through symlinks when it exists.
function _snopt_file_key(path::AbstractString)
    p = abspath(_snopt_trim(path))
    dir, base = dirname(p), basename(p)
    if isdir(dir)
        dir = try
            realpath(dir)
        catch
            dir
        end
    end
    return joinpath(dir, base)
end

# Whether names in `dir` that differ only in case refer to the same file, found
# by creating a probe file there. macOS and Windows volumes are usually case
# insensitive, but not always. If the probe cannot be created, assume they are
# (two such names are then rejected, which is the safe side).
function _snopt_dir_is_case_insensitive(dir::AbstractString)
    # Not `rand`: leave the global random stream alone.
    name = "NLPInterfaceCaseProbe_" * string(hash((time_ns(), getpid(), objectid(dir))); base=16)
    probe = joinpath(dir, name)
    try
        touch(probe)
    catch
        return true
    end
    try
        return isfile(joinpath(dir, lowercase(name)))
    catch
        return true
    finally
        rm(probe; force=true)
    end
end

function _snopt_same_output_file(a::AbstractString, b::AbstractString)
    a_, b_ = _snopt_trim(a), _snopt_trim(b)
    (isempty(a_) || isempty(b_)) && return false
    (_snopt_is_console_name(a_) || _snopt_is_console_name(b_)) && return false
    ka, kb = _snopt_file_key(a_), _snopt_file_key(b_)
    ka == kb && return true
    # Different names for one existing file (file symlinks, hard links).
    try
        ispath(a_) && ispath(b_) && samefile(a_, b_) && return true
    catch
    end
    # Names that differ only in case, on a case-insensitive volume.
    if (Sys.isapple() || Sys.iswindows()) && lowercase(ka) == lowercase(kb)
        dir = dirname(ka)
        return !isdir(dir) || _snopt_dir_is_case_insensitive(dir)
    end
    return false
end

# `test(path)` (`ispath`, `isdir`, ...), with `false` where `stat` itself fails,
# e.g. ELOOP on a symlink loop.
function _snopt_stat_ok(test, path)
    return try
        test(path)
    catch err
        err isa Base.IOError || rethrow()
        false
    end
end

# Open `path` for appending (creating it if needed) to find out whether SNOPT
# will be able to write it, then remove it again if it did not exist. Paths that
# exist but are not regular files (`/dev/null`, a FIFO) are not probed, since
# opening a FIFO blocks.
function check_snopt_output_writable(opt::AbstractString, path::AbstractString, depth::Int=0)
    if islink(path) && !_snopt_stat_ok(ispath, path)
        # A dangling symlink: opening it creates its target, so probe (and then
        # remove) the target instead of deleting the link. Give up on cycles.
        depth < 32 || throw(ArgumentError("$opt is a symlink loop: $(repr(path))"))
        target = readlink(path)
        isabspath(target) || (target = joinpath(dirname(abspath(path)), target))
        return check_snopt_output_writable(opt, target, depth + 1)
    end
    existed = _snopt_stat_ok(ispath, path)
    existed && !_snopt_stat_ok(isfile, path) && return nothing
    try
        open(identity, path, "a")
    catch err
        throw(
            ArgumentError(
                "$opt cannot be opened for writing: $(repr(path)) " *
                "($(sprint(showerror, err)))",
            ),
        )
    end
    existed || rm(path; force=true)
    return nothing
end

# Opening one file on both the Print and Summary units makes the Fortran OPEN
# fail; SNOPT then returns INFO 91 without solving, and the unit is left
# half-open so every later solve that names that file fails the same way until
# Julia restarts. Reject such pairs before SNOPT sees them. Also reject paths that
# cannot be opened for writing (missing directory, path is a directory, no write
# permission): SNOPT would silently fail on those, with INFO 91 or a misleading
# status such as `:Numerical_Difficulties`.
function check_snopt_output_files(printpath::AbstractString, summpath::AbstractString)
    try
        _check_snopt_output_files(printpath, summpath)
    catch err
        # e.g. EACCES from `stat` on a directory without search permission
        err isa Base.IOError || rethrow()
        throw(ArgumentError("Print_file/Summary_file cannot be used: $(sprint(showerror, err))"))
    end
    return nothing
end

function _check_snopt_output_files(printpath::AbstractString, summpath::AbstractString)
    for (opt, p) in (("Print_file", printpath), ("Summary_file", summpath))
        p_ = _snopt_trim(p)
        (isempty(p_) || _snopt_is_console_name(p_)) && continue
        full = abspath(p_)
        dir = dirname(full)
        isdir(dir) || throw(
            ArgumentError(
                "$opt directory does not exist: $(repr(dir)) " *
                "(paths are used literally; `~` is not expanded)",
            ),
        )
        _snopt_stat_ok(isdir, full) &&
            throw(ArgumentError("$opt names a directory, not a file: $(repr(full))"))
    end
    if _snopt_same_output_file(printpath, summpath)
        throw(
            ArgumentError(
                "Print_file and Summary_file refer to the same file " *
                "($(repr(printpath)) and $(repr(summpath))); SNOPT cannot open " *
                "one file on two output units. Use two different files " *
                "(names differing only in case are the same file on a " *
                "case-insensitive volume).",
            ),
        )
    end
    for (opt, p) in (("Print_file", printpath), ("Summary_file", summpath))
        p_ = _snopt_trim(p)
        (isempty(p_) || _snopt_is_console_name(p_)) && continue
        check_snopt_output_writable(opt, p_)
    end
    return nothing
end

function snopta_usrfun_trampoline(
    status_::Ptr{Cint},
    n_::Ptr{Cint},
    x_::Ptr{Cdouble},
    needF_::Ptr{Cint},
    neF_::Ptr{Cint},
    F_::Ptr{Cdouble},
    needG_::Ptr{Cint},
    neG_::Ptr{Cint},
    G_::Ptr{Cdouble},
    ::Ptr{UInt8},
    ::Ptr{Cint},
    ::Ptr{Cint},
    ::Ptr{Cint},
    ::Ptr{Cdouble},
    ::Ptr{Cint},
)::Cvoid
    try
        cb = _SNOPT_USRFUN[]
        cb === nothing && error("NLPInterface: SNOPT called back with no active solve")
        cb(status_, n_, x_, needF_, neF_, F_, needG_, neG_, G_)
    catch err
        status = snopt_callback_status(err)
        if status <= -2
            # Keep the first error: SNOPT stops after this callback returns.
            if _SNOPT_CALLBACK_ERROR[] === nothing
                _SNOPT_CALLBACK_ERROR[] = err
                _SNOPT_CALLBACK_BACKTRACE[] = catch_backtrace()
            end
        else
            _SNOPT_LAST_DOMAIN_ERROR[] = err
        end
        unsafe_store!(status_, status)
    end
    return
end

# Run `f` holding `SNOPT_LOCK`. If another solve holds it for longer than
# `SNOPT_LOCK_WAIT_WARN_SECONDS`, warn: a deadlock is likely when the running
# solve's `fitness` waits for this task (it cannot be detected reliably).
function with_snopt_lock(f)
    if !trylock(SNOPT_LOCK)
        delay = SNOPT_LOCK_WAIT_WARN_SECONDS[]
        timer = Timer(delay) do _
            # Once per session: concurrent long solves wait legitimately.
            @warn "solve_snopt has waited $(delay) s for another SNOPT solve to " *
                  "finish. Only one SNOPT solve runs at a time, so this is expected " *
                  "when several tasks solve concurrently; but if the running solve's " *
                  "fitness waits for this task (for example a worker pool fed from " *
                  "fitness), it is a deadlock." maxlog = 1
        end
        try
            lock(SNOPT_LOCK)
        finally
            close(timer)
        end
    end
    try
        return f()
    finally
        unlock(SNOPT_LOCK)
    end
end

function dense_jacobian_pattern(nF::Int, n::Int)
    neG = nF * n
    iGfun = Vector{Int32}(undef, max(neG, 1))
    jGvar = Vector{Int32}(undef, max(neG, 1))
    k = 1
    for j in 1:n
        for i in 1:nF
            iGfun[k] = i
            jGvar[k] = j
            k += 1
        end
    end
    return iGfun, jGvar, neG
end

"""
    solve_snopt(fitness, x0, x_lb, x_ub, n_ceq, n_cineq; kwargs...)

Solve a constrained minimization problem with SNOPT.

`fitness(x)` must return `(obj, ceq, cineq)` with `ceq == 0` and `cineq <= 0`.

Keyword `deriv` selects derivatives (`:forwarddiff` default, or `:finitediff`).
`Print_file` and `Summary_file` select SNOPT output files. A nonzero
`Major_print_level` with no `Summary_file` prints the SNOPT summary (major
iteration log) to the console. All other keyword arguments are passed through
as SNOPT options (underscores become spaces).

The lengths of `ceq` and `cineq` must equal `n_ceq` and `n_cineq` at every
evaluation; otherwise a `DimensionMismatch` is thrown. `x0` is not modified.

Bounds must satisfy `x_lb <= x_ub` and not be `NaN`, and `x0` must be
non-empty and finite; otherwise an `ArgumentError` is thrown.

`Print_file` and `Summary_file` must name different files that can be opened
for writing; otherwise an `ArgumentError` is thrown before SNOPT starts. Only
the exact name `"screen"` means the console; any other name, including
`"stdout"`, is a file. On libraries without `f_sninitx`, files are opened with
SNOPT's own Fortran helpers; if the library lacks those too, the file names are
ignored with a warning.

If `fitness` throws a `DomainError` (directly, or from a task it spawned, i.e.
wrapped in a `TaskFailedException` or `CompositeException`), SNOPT treats the
functions as undefined at that point and shortens the step; the error is not
rethrown. If SNOPT cannot recover, it ends with a failure status, usually
`:User_Supplied_Function_Undefined` (then the last `DomainError` is shown in a
warning). Any other exception, including `InterruptException` (Ctrl-C), stops
SNOPT at once and is rethrown by `solve_snopt`.

Only one SNOPT solve runs at a time. Calling `solve_snopt` from inside
`fitness` (or, on Julia 1.11 and later, from a task spawned there while the
solve runs) throws an error, since SNOPT's C interface keeps global state.
Handing the call to a task that existed before the solve deadlocks; a solve
that waits more than a minute for another one to finish logs a warning saying
so.

`fitness` must not keep a reference to `x`: the array is reused between calls
(copy it to keep it).

Returns `(xopt, fopt, info)`. `info` is a symbol from
`NLPInterface.SNOPT_RETURN_STATUS`. `fopt` is the objective of a plain
(`Float64`) evaluation of `fitness` at `xopt`, so it matches `fitness(xopt)`
even when SNOPT's last evaluation was at another point (a derivative check) or
used ForwardDiff Dual numbers; `fitness` is called once more after the solve
unless its last plain evaluation was at `xopt`. `fopt` is `NaN` if no
evaluation of `fitness` requested by SNOPT completed, or if `fitness` throws a
`DomainError` at `xopt`. `:Solve_Succeeded`,
`:Solved_To_Acceptable_Level`, and `:Feasible_Point_Found` (only with the
`Feasible point` option) mean SNOPT returned a feasible point.
`:Elastic_Objective_Minimized` and `:Elastic_Infeasibilities_Minimized` end the
run normally in elastic mode, where the manual does not promise the constraints
hold, so check them before using `xopt`.
"""
function solve_snopt(
    fitness::Function,
    x0::AbstractVector,
    x_lb::AbstractVector,
    x_ub::AbstractVector,
    n_ceq::Integer,
    n_cineq::Integer;
    deriv::Symbol=:forwarddiff,
    kwargs...,
)
    require_snopt()
    check_snopt_not_nested()
    check_solver_inputs(x0, x_lb, x_ub, n_ceq, n_cineq, deriv)
    n = length(x0)

    # Always a fresh array: SNOPT writes the solution into `x` in place, and
    # `convert` would return the caller's `x0` itself when it is a Vector{Float64}.
    x = Vector{Float64}(x0)
    xlow = snopt_bound.(x_lb)
    xupp = snopt_bound.(x_ub)
    nF = 1 + Int(n_ceq) + Int(n_cineq)
    Flow = fill(-SNOPT_INF, nF)
    Fupp = fill(SNOPT_INF, nF)
    if n_ceq > 0
        Flow[2:(1 + n_ceq)] .= 0.0
        Fupp[2:(1 + n_ceq)] .= 0.0
    end
    if n_cineq > 0
        Flow[(2 + n_ceq):end] .= -SNOPT_INF
        Fupp[(2 + n_ceq):end] .= 0.0
    end

    iGfun, jGvar, neG = dense_jacobian_pattern(nF, n)
    iAfun = Int32[1]
    jAvar = Int32[1]
    Aval = [0.0]
    neA = 0

    F = zeros(Float64, nF)
    xstate = zeros(Int32, n)
    Fstate = zeros(Int32, nF)
    xmul = zeros(Float64, n)
    Fmul = zeros(Float64, nF)

    evaluator = ResidualEvaluator(fitness, n, n_ceq, n_cineq, deriv)
    r_buf = zeros(Float64, nF)
    J_buf = zeros(Float64, nF, n)
    # Set once `fitness` has returned a value. Until then `F` holds only the
    # zeros it was initialized with, which must not be reported as `fopt`.
    f_evaluated = Ref(false)
    # The last point whose residual came from a plain (Float64) call, and its
    # objective, so the final `fopt` can skip a call when SNOPT ends there.
    x_plain = fill(NaN, n)
    f_plain = Ref(NaN)

    function usrfun(
        status_::Ptr{Cint},
        n_::Ptr{Cint},
        x_::Ptr{Cdouble},
        needF_::Ptr{Cint},
        neF_::Ptr{Cint},
        F_::Ptr{Cdouble},
        needG_::Ptr{Cint},
        neG_::Ptr{Cint},
        G_::Ptr{Cdouble},
    )
        status = unsafe_load(status_)
        status >= 2 && return
        nx = Int(unsafe_load(n_))
        xv = unsafe_wrap(Array, x_, nx)
        needF = unsafe_load(needF_) > 0
        needG = unsafe_load(needG_) > 0
        if needG
            # One pass gives both the value and the Jacobian (see ResidualEvaluator).
            residual_value_and_jacobian!(evaluator, r_buf, J_buf, xv, false)
            Gg = unsafe_wrap(Array, G_, Int(unsafe_load(neG_)))
            Gg .= vec(J_buf)
        elseif needF
            residual_value!(evaluator, r_buf, xv)
        else
            return
        end
        f_evaluated[] = true
        if !needG || deriv === :finitediff
            # The residual came from a plain call (finite differences evaluate
            # the base point plainly).
            x_plain .= xv
            f_plain[] = r_buf[1]
        end
        if needF
            Ff = unsafe_wrap(Array, F_, Int(unsafe_load(neF_)))
            Ff .= r_buf
        end
        return
    end

    printfile = ""
    summfile = ""
    major_print_level = nothing
    solver_kwargs = Pair{Any,Any}[]
    for (key, val) in kwargs
        keyword = snopt_option_keyword(key)
        if keyword == "Print file"
            printfile = string(val)
        elseif keyword == "Summary file"
            summfile = string(val)
        else
            keyword == "Major print level" && (major_print_level = val)
            push!(solver_kwargs, key => val)
        end
    end
    printpath, summpath = snopt_output_files(
        printfile,
        summfile;
        console_summary=_snopt_console_summary(major_print_level, summfile),
    )
    if libsnopt7_has_sninitx || snopt_fallback_opens_files()
        check_snopt_output_files(printpath, summpath)
    else
        is_file_name(p) = !isempty(_snopt_trim(p)) && !_snopt_is_console_name(p)
        if any(is_file_name, (printpath, summpath))
            @warn "This SNOPT library has neither f_sninitx nor snFileOpenWrite; " *
                  "Print_file and Summary_file names are ignored (only \"screen\" " *
                  "is available)." maxlog = 1
        end
        is_file_name(printpath) && (printpath = "")
        is_file_name(summpath) && (summpath = "")
    end

    iw = zeros(Int32, 500 + 100 * (n + nF))
    rw = zeros(Float64, 500 + 200 * (n + nF))
    cw = zeros(UInt8, 500 * 8)
    iu = Int32[0]
    ru = [0.0]
    initialized = Ref(false)
    fallback_units = Cint[]

    inform = Ref{Cint}(0)
    nS = Ref{Cint}(0)
    nInf = Ref{Cint}(0)
    sInf = Ref{Cdouble}(0)
    miniw = Ref{Cint}(0)
    minrw = Ref{Cint}(0)

    usr_callback = @cfunction(
        snopta_usrfun_trampoline,
        Cvoid,
        (
            Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
            Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{UInt8}, Ptr{Cint}, Ptr{Cint},
            Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint},
        ),
    )

    return with_snopt_lock() do
        _with_snopt_in_solve() do
            _SNOPT_USRFUN[] = usrfun
            _SNOPT_CALLBACK_ERROR[] = nothing
            _SNOPT_LAST_DOMAIN_ERROR[] = nothing
            try
                # Units opened for the fallback path are closed in `finally`
                # even if this throws.
                snopt_init_workspace!(printpath, summpath, cw, iw, rw, fallback_units)
                initialized[] = true

                meminfo = Ref{Cint}(0)
                ccall(
                    (:f_snmema, libsnopt7),
                    Cvoid,
                    (
                        Ptr{Cint}, Cint, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint},
                        Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                    ),
                    meminfo,
                    nF,
                    n,
                    neA,
                    neG,
                    miniw,
                    minrw,
                    iw,
                    length(iw),
                    rw,
                    length(rw),
                )
                if meminfo[] != SNOPT_MEMA_OK
                    # `miniw`/`minrw` are not meaningful; report SNOPT's status
                    # (81-84: not enough workspace) without solving.
                    return copy(x), NaN, get(SNOPT_RETURN_STATUS, Int(meminfo[]), :Unknown)
                end
                if Int(miniw[]) > length(iw)
                    resize!(iw, Int(miniw[]))
                    snopt_set_option!(iw, length(iw), rw, length(rw), "Total integer workspace", length(iw))
                end
                if Int(minrw[]) > length(rw)
                    resize!(rw, Int(minrw[]))
                    snopt_set_option!(iw, length(iw), rw, length(rw), "Total real workspace", length(rw))
                end

                for (key, val) in solver_kwargs
                    apply_snopt_option!(iw, length(iw), rw, length(rw), key, val)
                end

                GC.@preserve usrfun F x xlow xupp Flow Fupp xstate Fstate xmul Fmul iAfun jAvar Aval iGfun jGvar iw rw cw iu ru begin
                    ccall(
                        (:f_snopta, libsnopt7),
                        Cvoid,
                        (
                            Cint, Cstring,
                            Cint, Cint, Cdouble, Cint,
                            Ptr{Cvoid},
                            Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cdouble},
                            Ptr{Cint}, Ptr{Cint}, Cint,
                            Ptr{Cdouble}, Ptr{Cdouble},
                            Ptr{Cdouble}, Ptr{Cdouble},
                            Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble},
                            Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble},
                            Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                            Ptr{Cint}, Ptr{Cint},
                            Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                            Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
                        ),
                        Cint(0),
                        "NLPIntf",
                        nF,
                        n,
                        0.0,
                        1,
                        usr_callback,
                        iAfun,
                        jAvar,
                        neA,
                        Aval,
                        iGfun,
                        jGvar,
                        neG,
                        xlow,
                        xupp,
                        Flow,
                        Fupp,
                        x,
                        xstate,
                        xmul,
                        F,
                        Fstate,
                        Fmul,
                        inform,
                        nS,
                        nInf,
                        sInf,
                        miniw,
                        minrw,
                        iu,
                        length(iu),
                        ru,
                        length(ru),
                        iw,
                        length(iw),
                        rw,
                        length(rw),
                    )
                end
                if _SNOPT_CALLBACK_ERROR[] !== nothing
                    err = _SNOPT_CALLBACK_ERROR[]
                    _SNOPT_CALLBACK_ERROR[] = nothing
                    @debug "fitness threw inside solve_snopt; backtrace at the throw site:" exception = (err, _SNOPT_CALLBACK_BACKTRACE[])
                    throw(err)
                end
                status = Int(inform[])
                info = get(SNOPT_RETURN_STATUS, status, :Unknown)
                domain_err = _SNOPT_LAST_DOMAIN_ERROR[]
                if domain_err !== nothing && 61 <= status <= 63
                    @warn "SNOPT stopped at a point where fitness is undefined (INFO $status); " *
                          "the last DomainError thrown by fitness was:" exception = domain_err
                end
                # NaN if no `fitness` evaluation requested by SNOPT completed (e.g.
                # INFO 91, or INFO 61 when `fitness` is undefined at `x0`). `F[1]`
                # is not used: SNOPT may leave it at another point (its derivative
                # check perturbs `x`), and with ForwardDiff it comes from a Dual
                # evaluation.
                fopt = if !f_evaluated[]
                    NaN
                elseif isequal(x_plain, x)
                    f_plain[]
                else
                    try
                        residual_value!(evaluator, r_buf, x)[1]
                    catch err
                        is_domain_error(err) || rethrow()
                        NaN
                    end
                end
                return copy(x), fopt, info
            finally
                _SNOPT_USRFUN[] = nothing
                _SNOPT_CALLBACK_ERROR[] = nothing
                _SNOPT_CALLBACK_BACKTRACE[] = nothing
                _SNOPT_LAST_DOMAIN_ERROR[] = nothing
                if initialized[]
                    ccall(
                        (:f_snend, libsnopt7),
                        Cvoid,
                        (Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
                        iw,
                        length(iw),
                        rw,
                        length(rw),
                    )
                end
                snopt_close_fallback_units!(fallback_units)
            end
        end
    end
end
