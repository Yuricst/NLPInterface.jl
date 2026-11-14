using Base.ScopedValues
const sv = ScopedValue(0)
for i in 1:200
    t = with(() -> @async(with(() -> GC.gc(true), sv => 2)), sv => 1)
    wait(t)
    GC.gc(true)
end
println("done")
