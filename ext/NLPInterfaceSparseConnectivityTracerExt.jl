module NLPInterfaceSparseConnectivityTracerExt

using SparseConnectivityTracer: TracerSparsityDetector

# Used by `jac_sparsity=:auto` and `detect_jac_sparsity` without `detector`.
default_detector() = TracerSparsityDetector()

end # module
