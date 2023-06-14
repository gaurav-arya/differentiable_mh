using LinearAlgebra
using Distributions
using GLMakie
import Random
using Colors
d(θ) = Normal(θ, 1)
σ = 2.38
proposal(x) = Normal(x, σ)
P = proposal
q(x, x_prop) = pdf(proposal(x), x_prop)
qrand(x) = rand(proposal(x))
Random.seed!(1)

θ = 1.0
K = 6
x = 0.0
xs = Float64[]
xos = Float64[]
accs = Bool[]
let x = x; for k in 1:K
    xᵒ = rand(P(x))
    push!(xos, xᵒ)
    lα = (logpdf(d(θ), xᵒ) - logpdf(d(θ), x) + logpdf(P(xᵒ), x) - logpdf(P(x), xᵒ))
    if -randexp() < lα
        x = xᵒ
        acc = true
    else 
        acc = false
    end
    push!(accs, acc)
    push!(xs, x)
end end
Random.seed!(1)
 
x = 0.0
xs2 = Float64[]
xos2 = Float64[]
accs2 = Bool[]
let x = x; for k in 1:K
    xᵒ = rand(P(x))
    push!(xos2, xᵒ)
    lα = (logpdf(d(θ), xᵒ) - logpdf(d(θ), x) + logpdf(P(xᵒ), x) - logpdf(P(x), xᵒ))
    acc =  -randexp() < lα
    if k == 3
        acc = !acc
    end
    if acc 
        x = xᵒ
    end
    push!(accs2, acc)
    push!(xs2, x)
end end


fig = begin
k = 0
accepted = Observable(xs[1:k])
oldproposed = Observable(xos[1:k])
newproposed = Observable([Point2(k+1, xos[k+1])])
accepted2 = Observable(Point2.(zip(1:k, xs2[1:k])))
oldproposed2 = Observable(Point2.(zip(1:k, xos2[1:k])))
newproposed2 = Observable([Point2(k+1, xos[k+1])][1:0])

f = Figure(resolution=(400, 300))
ax = Axis(f[1,1], xlabel="t", ylabel="Sample")
xlims!(ax, 0, K+4) 
ylims!(ax, -6.1, 6.1) 
c = colorant"#0072b3"
f1 = scatterlines!(ax, accepted, markercolor=c, color=c, markersize=15.0)
scatter!(ax, oldproposed, color = (:white, 0.0), markersize=12, strokewidth=1.3, strokecolor=(c,0.3))
scatter!(ax, newproposed, color = (:white, 0.0), markersize=12, strokewidth=1.3, strokecolor=c)
c = colorant"#730000"
f2 = scatterlines!(ax, accepted2, markercolor=(c,0.5), color=(c,0.5), markersize=15.0)
scatter!(ax, oldproposed2, color = (:white, 0.0), markersize=12, strokewidth=1.3, strokecolor=(c,0.3))
scatter!(ax, newproposed2, color = (:white, 0.0), markersize=12, strokewidth=1.3, strokecolor=c)
Legend(f[1, 1],
    [f1, f2],
    ["original", "perturbed"],
    tellheight = false,
    tellwidth = false,
    margin = (10, 10, 10, 10),
    halign = 1, valign = 1,
    )

f
end

states = [(k, s) for  s in 1:2, k in 1:K]

record(fig, "mh.gif", states; framerate = 1) do st
    k, s = st 
    if s == 1
        newproposed[] = [Point2(k, xos[k])]
        if k >= 3 
            newproposed2[] = [Point2(k, xos2[k])]
        end
    else
        accepted[] = xs[1:k]
        oldproposed[] = xos[1:k]
        accepted2[] = (Point2.(zip(1:k, xs2[1:k])))
        oldproposed2[] =  (Point2.(zip(2:k, xos2[2:k])))
    end
end
