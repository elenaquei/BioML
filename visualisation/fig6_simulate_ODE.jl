using DifferentialEquations, MAT, Random, RecursiveArrayTools, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, Optimisers, OptimizationOptimisers, Plots

# we load both the optimization file as well as the optimized parameter values constructed using optimise_ODE.jl
netfile = matread("damaged_opt_data.mat")

paramfile = matread("opts_hill/opt_hill_damaged_4.mat")

# load in data from output files (activating network, inhibiting network and expression of all genes)
const Aa_ref = netfile["Aa"]
const Ai_ref = netfile["Ai"]
const A_ref = Aa_ref + Ai_ref
const data0  = netfile["data0"]
const data1 = netfile["data1"]
const data = [data0,data1]

const N = length(Aa_ref[:,1])
const xsize = sum(Aa_ref.>0) + sum(Ai_ref.>0)

bestp = paramfile["pbest"]

nsims = 2
initial_conditions = [
    abs.(bestp[1:N]),
    abs.(bestp[N+1:2*N])
]

# set constants (amount of genes + amount of parameters)
const N = length(Aa_ref[:,1])
const xsize = sum(Aa_ref.>0) + sum(Ai_ref.>0)

# set time span to simulate and time steps on which data is available
tspan =  (0.1f0, 1.0f0)
const tsteps = 0.0f0:0.1f0:0.9f0

# function to recover adjacency matrices from input data
function make_matrices(x)

    idxs = findall(!iszero, A_ref)
    Aa_vals = map(xi -> max(xi, 0.0), x)
    Ai_vals = map(xi -> max(-xi, 0.0), x)

    # use comprehensions to fill in the adjacency matrices. Notably, this definition does not mutate arrays, and therefore this can also be used in the case that automatic differentiation is desired
    Aa_new = [i in idxs ? Aa_vals[findfirst(isequal(i), idxs)] : 0.0 for i in CartesianIndices(A_ref)]
    Ai_new = [i in idxs ? Ai_vals[findfirst(isequal(i), idxs)] : 0.0 for i in CartesianIndices(A_ref)]

    return reshape(Aa_new, size(A_ref)), reshape(Ai_new, size(A_ref))
end

# ODE function definition
function grnODE(du, u, p, t)
    # set parameters: τ is the time scale, p is distributed over the activating and inhibiting matrices, and e is internal stimulation (currently unused!)
    e = abs.(p[length(p)-N:length(p)-1])
    τ = abs.(p[length(p)])
    param = p[1:length(p)-N-1]
    
    # make matrices and compute ODE function at given point u
    (Aa,Ai) = make_matrices(param)

    act(x,y) = x ./ (1 .+ x .+ y)
    du[:] = τ*(-u + act.(e + transpose(Aa)*u.^3,transpose(Ai)*u.^3))
end

# solve ODE for both initial conditions, and then plot in a phase portrait
prob = ODEProblem(grnODE,initial_conditions[1],(0.0f0,0.9f0),bestp[2*N+1:end])
sol = solve(prob, Tsit5())
u1 = reduce(hcat, sol.u)

p = plot(u1[4,:],u1[5,:], label="Simulation from IC 1", color=:blue, linewidth=3)

prob2 = ODEProblem(grnODE,initial_conditions[2],(0.0f0,0.9f0),bestp[2*N+1:end])
sol2 = solve(prob2, Tsit5())
u2 = reduce(hcat, sol2.u)

plot!(u2[4,:],u2[5,:], label="Simulation from IC 2", color=:red, linewidth=3)

plot!(data0[4,:],data0[5,:],color=:black, linestyle=:dash, linewidth=3)
plot!(data1[4,:],data1[5,:],label="Data",color=:black, linestyle=:dash, linewidth=3)

fig_size = (300, 300)

# phase portrait
plt = plot(u1[4, :], u1[5, :],
    linewidth = 2,
    linecolor = :blue,
    xlabel = "Gene 4 expression (a.u.)",
    ylabel = "Gene 5 expression (a.u.)",
    legend = false,
    size = fig_size,
    tickfont = font(10),
    guidefont = font(10),
    framestyle = :box,
    aspect_ratio = :equal)

# plot second trajectory
plot!(plt, u2[4, :], u2[5, :],
    linewidth = 2,
    linecolor = :red)

# add arrows to indicate direction
function add_arrows(u; color=:black)
    for i in 2:2:(size(u, 2)-1)
        x = u[4, i]
        y = u[5, i]
        dx = u[4, i+1] - u[4, i]
        dy = u[5, i+1] - u[5, i]
        scale = 0.001
        quiver!([x], [y],
            quiver=([dx * scale], [dy * scale]),
            color=color,
            arrow=true,
            linewidth=3)
    end
end

add_arrows(u1, color=:blue)
add_arrows(u2, color=:red)

# plot input data
plot!(data0[4,:],data0[5,:],color=:black, linestyle=:dash, linewidth=3)
plot!(data1[4,:],data1[5,:],color=:black, linestyle=:dash, linewidth=3)

add_arrows(data0, color=:black)
add_arrows(data1, color=:black)
xlims!(plt, 0.0, 1.0)
ylims!(plt, 0.0, 1.0)

display(plt)
