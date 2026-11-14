module NLPInterface

using DiffResults
using ForwardDiff
using Ipopt
using Libdl
import ADTypes
import SparseArrays
import SparseMatrixColorings

export solve_ipopt, solve_snopt, has_snopt, detect_jac_sparsity, NLPResult

include("validation.jl")
include("result.jl")
include("residuals.jl")
include("sparsity.jl")
include("ipopt.jl")
include("snopt.jl")

function __init__()
    global libsnopt7 = find_snopt_lib()
    return nothing
end

end # module NLPInterface
