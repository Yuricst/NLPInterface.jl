# Minimal reproducer (Julia only, no packages). A task created inside
# `with(SV => x)` runs `with(SV => y)` and a full GC runs inside it.
#
#   julia --heap-size-hint=300M repro/mwe.jl VARIANT [N]
#   VARIANT = spawn_ref | spawn_int | async_ref | spawn_ref_sametype | direct_ref
using Base.ScopedValues

variant = ARGS[1]
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
println("mwe.jl: ", variant, " N=", N, " julia ", VERSION, " threads ",
    Threads.nthreads(), "+", Threads.nthreads(:interactive))
flush(stdout)

const SVANY = ScopedValue{Any}(nothing)
const SVREF = ScopedValue{Union{Nothing,Base.RefValue{Bool}}}(nothing)

function once(variant)
    if variant == "spawn_ref"          # value is a fresh Ref, ScopedValue{Any}
        t = with(SVANY => Ref(true)) do
            Threads.@spawn with(() -> GC.gc(true), SVANY => Ref(true))
        end
        wait(t)
    elseif variant == "spawn_int"      # value is an Int
        t = with(SVANY => 1) do
            Threads.@spawn with(() -> GC.gc(true), SVANY => 2)
        end
        wait(t)
    elseif variant == "async_ref"      # same thread, @async
        t = with(SVANY => Ref(true)) do
            @async with(() -> GC.gc(true), SVANY => Ref(true))
        end
        wait(t)
    elseif variant == "spawn_ref_sametype"  # NLPInterface's ScopedValue type
        t = with(SVREF => Ref(true)) do
            Threads.@spawn with(() -> GC.gc(true), SVREF => Ref(true))
        end
        wait(t)
    elseif variant == "direct_ref"     # no task: nested with + GC in the same task
        with(SVANY => Ref(true)) do
            with(() -> GC.gc(true), SVANY => Ref(true))
        end
    else
        error("unknown variant $variant")
    end
end

for i in 1:N
    once(variant)
    GC.gc(true)
end
println("mwe.jl: completed ", N, " iterations")
