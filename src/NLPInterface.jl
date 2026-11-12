module NLPInterface

using DiffResults
using ForwardDiff
using Ipopt
using Libdl

export solve_ipopt, solve_snopt, has_snopt

include("validation.jl")
include("residuals.jl")
include("ipopt.jl")
include("snopt.jl")

function __init__()
    global libsnopt7 = find_snopt_lib()
    return nothing
end

end # module NLPInterface
