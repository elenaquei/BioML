using DifferentialEquations, MAT, Random, RecursiveArrayTools, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, Optimisers, OptimizationOptimisers, Plots

# we load both the optimization file as well as the optimized parameter values constructed using optimise_ODE.jl
netfile = matread("gae_new_opt_data.mat")

paramfile = matread("opts_hill/opt_hill_gae_1.mat")

# load in data from output files (activating network, inhibiting network and expression of all genes)
const Aa_ref = netfile["Aa"]
const Ai_ref = netfile["Ai"]
const A_ref = Aa_ref + Ai_ref

# here, 2 trajectories are loaded (this can be adjusted by hand)
const data0  = netfile["data0"]
const data1 = netfile["data1"]

# data is constructed from the two input files manually (this can be adjusted if a different setting with more/fewer trajectories is used)
const data = [data0,data1]

# set constants (amount of genes + amount of parameters)
const N = length(Aa_ref[:,1])
const xsize = sum(Aa_ref.>0) + sum(Ai_ref.>0)

bestp = paramfile["pbest"]

# again, number of simulations can be set manually
nsims = 2
initial_conditions = [
    abs.(bestp[1:N]),
    abs.(bestp[N+1:2*N])
]

# set time span to simulate and time steps on which data is available (SET MANUALLY)
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
    # set parameters: τ is the time scale, p is distributed over the activating and inhibiting matrices, and e is internal stimulation
    e = abs.(p[length(p)-N:length(p)-1])
    τ = abs.(p[length(p)])
    param = p[1:length(p)-N-1]
    
    # make matrices and compute ODE function at given point u
    (Aa,Ai) = make_matrices(param)

    # we explicitly increase the degradation rate of gene 3
    act(x,y) = x ./ (1 .+ x .+ y)
    du[:] = τ*(-u + act.(e + transpose(Aa)*u.^3,transpose(Ai)*u.^3))
    du[3] = τ*(-3*u + act.(e + transpose(Aa)*u.^3,transpose(Ai)*u.^3))[3]
end

# solve KD simulation
prob = ODEProblem(grnODE,initial_conditions[1],(0.0f0,0.9f0),bestp[2*N+1:end])
sol = solve(prob, Tsit5())
u1 = reduce(hcat, sol.u)

# plot results
p = plot(sol, vars=5, label="Simulation", color=:blue, linewidth=3)

# if reference data is available for the kd experiment, this is loaded here
kddata = matread("kd_data.mat")
kd_ts = kddata["kddat"]

plot!(tsteps, kd_ts, label="True model", color=:black, linestyle=:dash, linewidth=3)

plot!(xlabel="Pseudotime", ylabel="Gene 5 expression (a.u.)",
      legendfontsize=12, tickfontsize=10,
      guidefontsize=10, titlefontsize=16, legend=false)

plot!(size=(400, 300), ylims=(0, 1))
