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
const SNOPT_TEMPDIRS = String[]
const SNOPT_OPEN_HANDLES = Ptr{Cvoid}[]

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
# SNOPT immediately; `solve_snopt` then rethrows the error. Only a `DomainError`
# thrown directly counts: one wrapped in another exception (for example a
# `TaskFailedException` from a task spawned in `fitness`) stops the solve.
snopt_callback_status(err) = err isa DomainError ? Cint(-1) : Cint(-2)

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

function relocate_snopt_library(libpath::AbstractString)
    Sys.isapple() || return String(libpath)
    missing_deps = snopt_missing_install_names(libpath)
    isempty(missing_deps) && return String(libpath)
    dir = dirname(abspath(libpath))
    core = joinpath(dir, "libsnopt7." * Libdl.dlext)
    isfile(core) || return String(libpath)
    tool = Sys.which("install_name_tool")
    tool === nothing && return String(libpath)
    tmp = mktempdir()
    push!(SNOPT_TEMPDIRS, tmp)
    tmpcore = joinpath(tmp, "libsnopt7." * Libdl.dlext)
    tmpwrap = joinpath(tmp, basename(libpath))
    cp(core, tmpcore; follow_symlinks=true)
    cp(libpath, tmpwrap; follow_symlinks=true)
    for old in missing_deps
        try
            run(pipeline(`$tool -change $old @loader_path/libsnopt7.$(Libdl.dlext) $tmpwrap`; stderr=devnull))
        catch
            return String(libpath)
        end
    end
    codesign = Sys.which("codesign")
    if codesign !== nothing
        try
            run(pipeline(`$codesign --force --sign - $tmpwrap`; stdout=devnull, stderr=devnull))
            run(pipeline(`$codesign --force --sign - $tmpcore`; stdout=devnull, stderr=devnull))
        catch
            return String(libpath)
        end
    end
    return tmpwrap
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
        prepared = relocate_snopt_library(libpath)
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
    push!(dirs, joinpath(pkgroot, "__references", "libsnopt7_cpp"))
    push!(dirs, joinpath(pkgroot, "deps", "usr", "lib"))
    if Sys.iswindows()
        haskey(ENV, "PATH") && append!(dirs, split(ENV["PATH"], ';'))
    else
        haskey(ENV, "LD_LIBRARY_PATH") && append!(dirs, split(ENV["LD_LIBRARY_PATH"], ':'))
        if Sys.isapple() && haskey(ENV, "DYLD_LIBRARY_PATH")
            append!(dirs, split(ENV["DYLD_LIBRARY_PATH"], ':'))
        end
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

function snopt_init_workspace!(printpath, summpath, cw, iw, rw)
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
        # `sninit_` in the companion libsnopt7 library is reliable.
        if isempty(libsnopt7_core)
            global libsnopt7_core = libsnopt7
        end
        iprint = Ref{Cint}(0)
        # Unit 6 is Fortran stdout (SNOPT Summary file; see snInit / Major print level).
        isumm = Ref{Cint}(_snopt_is_screen(summpath) ? 6 : 0)
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

function _snopt_is_screen(path::AbstractString)
    s = lowercase(strip(String(path)))
    return s == "screen" || s == "stdout" || s == "-"
end

function _snopt_console_summary(major_print_level, summfile::AbstractString)
    isempty(summfile) || return _snopt_is_screen(summfile)
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

# f_sninitx treats only the exact word "screen" (after trimming) as the console.
_snopt_is_console_name(path::AbstractString) = _snopt_trim(path) == "screen"

# Normalized key for comparing output file names: absolute path with the
# directory resolved through symlinks when it exists. Errs on the side of calling
# two names the same: case is folded on macOS/Windows, whose file systems are
# case-insensitive by default.
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
    key = joinpath(dir, base)
    return (Sys.isapple() || Sys.iswindows()) ? lowercase(key) : key
end

function _snopt_same_output_file(a::AbstractString, b::AbstractString)
    a_, b_ = _snopt_trim(a), _snopt_trim(b)
    (isempty(a_) || isempty(b_)) && return false
    (_snopt_is_console_name(a_) || _snopt_is_console_name(b_)) && return false
    _snopt_file_key(a_) == _snopt_file_key(b_) && return true
    # Different names for one existing file (file symlinks, hard links).
    return try
        ispath(a_) && ispath(b_) && samefile(a_, b_)
    catch
        false
    end
end

# Opening one file on both the Print and Summary units makes the Fortran OPEN
# fail; SNOPT then returns INFO 91 without solving, and the unit is left
# half-open so every later solve that names that file fails the same way until
# Julia restarts. Reject such pairs before SNOPT sees them. Also reject paths that
# cannot be opened for an obvious reason (missing directory, path is a
# directory): SNOPT would silently return INFO 91 for those. Other OPEN failures,
# such as a directory without write permission, still surface only as
# `:Invalid_Problem_Definition`.
function check_snopt_output_files(printpath::AbstractString, summpath::AbstractString)
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
        isdir(full) && throw(ArgumentError("$opt names a directory, not a file: $(repr(full))"))
    end
    if _snopt_same_output_file(printpath, summpath)
        throw(
            ArgumentError(
                "Print_file and Summary_file refer to the same file " *
                "($(repr(printpath)) and $(repr(summpath))); SNOPT cannot open " *
                "one file on two output units. Use two different files " *
                "(on macOS/Windows, names differing only in case count as the same).",
            ),
        )
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
            _SNOPT_CALLBACK_ERROR[] === nothing && (_SNOPT_CALLBACK_ERROR[] = err)
        else
            _SNOPT_LAST_DOMAIN_ERROR[] = err
        end
        unsafe_store!(status_, status)
    end
    return
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

`Print_file` and `Summary_file` must name different files; two names for one
file throw an `ArgumentError`.

If `fitness` throws a `DomainError`, SNOPT treats the functions as undefined at
that point and shortens the step; the error is not rethrown. If SNOPT cannot
recover, it ends with a failure status, usually
`:User_Supplied_Function_Undefined` (then the last `DomainError` is shown in a
warning). Any other exception, including `InterruptException` (Ctrl-C) and a
`DomainError` wrapped in another exception, stops SNOPT at once and is rethrown
by `solve_snopt`.

Only one SNOPT solve runs at a time. Calling `solve_snopt` from inside
`fitness` (or, on Julia 1.11 and later, from a task spawned there while the
solve runs) throws an error, since SNOPT's C interface keeps global state.
Handing the call to a task that existed before the solve deadlocks.

`fitness` must not keep a reference to `x`: the array is reused between calls
(copy it to keep it).

Returns `(xopt, fopt, info)`. `info` is a symbol from
`NLPInterface.SNOPT_RETURN_STATUS`. `fopt` is `NaN` if no evaluation of
`fitness` requested by SNOPT completed. `:Solve_Succeeded`,
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
    n = length(x0)
    if n == 0
        throw(ArgumentError("x0 must be non-empty"))
    end
    if !(n == length(x_lb) == length(x_ub))
        throw(ArgumentError("x0, x_lb, and x_ub must have the same length"))
    end
    if n_ceq < 0 || n_cineq < 0
        throw(ArgumentError("n_ceq and n_cineq must be non-negative"))
    end
    if deriv !== :forwarddiff && deriv !== :finitediff
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end

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
        end
        if needF
            Ff = unsafe_wrap(Array, F_, Int(unsafe_load(neF_)))
            Ff .= r_buf
            f_evaluated[] = true
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
    if libsnopt7_has_sninitx
        check_snopt_output_files(printpath, summpath)
    elseif any(p -> !isempty(strip(p)) && !_snopt_is_screen(p), (printpath, summpath))
        @warn "This SNOPT library has no f_sninitx; Print_file and Summary_file " *
              "names are ignored (only a console summary is available)." maxlog = 1
    end

    iw = zeros(Int32, 500 + 100 * (n + nF))
    rw = zeros(Float64, 500 + 200 * (n + nF))
    cw = zeros(UInt8, 500 * 8)
    iu = Int32[0]
    ru = [0.0]
    initialized = Ref(false)

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

    return lock(SNOPT_LOCK) do
        _with_snopt_in_solve() do
            _SNOPT_USRFUN[] = usrfun
            _SNOPT_CALLBACK_ERROR[] = nothing
            _SNOPT_LAST_DOMAIN_ERROR[] = nothing
            try
                snopt_init_workspace!(printpath, summpath, cw, iw, rw)
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
                # INFO 91, or INFO 61 when `fitness` is undefined at `x0`).
                fopt = f_evaluated[] ? F[1] : NaN
                return copy(x), fopt, info
            finally
                _SNOPT_USRFUN[] = nothing
                _SNOPT_CALLBACK_ERROR[] = nothing
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
            end
        end
    end
end
