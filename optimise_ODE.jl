using DifferentialEquations
using Optimization, OptimizationOptimisers
using Zygote
using SciMLSensitivity
using ComponentArrays
using MAT
using Plots
using OptimizationBBO

# load in data for optimisation. This file should contain three fields:
#   Aa: reference activating adjacency matrix
#   Ai: reference inhibiting adjacency matrix
#   data: time series data to which model is fitted
# notably: Aa and Ai are combined to form one adjacency matrix in this script, and therefore are not used separately. Moreover, multiple data files can be given as input,
# which correspond to different trajectories in the data. Please refer to the parameter settings below to set this up for a different experiment.
#
# to save the output, the folder opts/ should exist in the working directory
netfile = matread("gae_new_opt_data.mat")

# load in data from output files (activating network, inhibiting network and expression of all genes)
const Aa_ref = netfile["Aa"]
const Ai_ref = netfile["Ai"]
const A_ref = Aa_ref + Ai_ref

# here, 2 trajectories are loaded (this can be adjusted by hand)
const data0  = netfile["data0"]
const data1 = netfile["data1"]

# data is constructed from the two input files manually (this can be adjusted if a different setting with more/fewer trajectories is used)
const data = [data0,data1]

const N = length(Aa_ref[:,1])
const xsize = sum(Aa_ref.>0) + sum(Ai_ref.>0)

# set time span to simulate and time steps on which data is available (SET MANUALLY)
tspan =  (0.1f0, 1.0f0)
const tsteps = 0.1f0:0.1f0:0.9f0

# set number of simulations (according to data available)
const nsims = 2

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

    # compute right-hand side of ODE
    act(x,y) = x ./ (1 .+ x .+ y)
    du[:] = τ*(-u + act.(e + transpose(Aa)*u.^3,transpose(Ai)*u.^3))
end

# loss function
function loss_function(θ,_)

    # distribute parameters θ into initial conditions and model parameters
    ics = reshape(abs.(θ[1:nsims*N]), (N, nsims))
    p = θ[nsims*N+1:end]

    l = 0

    # first, we compute the L2 loss between data an model simulations
    for k=1:nsims
        prob = ODEProblem(grnODE,ComponentVector(p=ics[:,k]),tspan,ComponentVector(p = p))
        sol = solve(prob, Tsit5(),saveat=tsteps, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        u = reduce(hcat,sol.u)
        l += sum(abs2, u .- data[k][:,2:end])
    end
    
    # then, we add L1 difference between current initial conditions and the desired ICs from the data
    l += 10*maximum(abs.(ics[:,1] .- data[1][:,1]))
    l += 10*maximum(abs.(ics[:,2] .- data[2][:,1]))

    # finally, we add the L1 loss on all parameter values (not initial conditions)
    l += 0.001*sum(abs.(p))
    return l
end

callback = function (state, l; doplot = false) #callback function to observe training

    # keep track of the amount of iterations already optimized for
    global iters += 1
    if iters % 10000 == 0
        display(string(iters)*" iterations...")
    end

    # since differential evolution takes many steps (many of which do not improve the objective), we only print out the loss value if it has improved
    if l < minimum(loss_vec)
        display(l)
    end

    # push current loss value to loss vector
    push!(loss_vec, l)
    return false
end


# this code runs 10 individual fits, but can be adjusted as needed
for k=1:10

    # construct optimization function
    optf = OptimizationFunction(loss_function, Optimization.AutoZygote())

    # set initial value for parameters
    θ0 = rand(N*nsims+xsize+N+1)
    θ0[1:N] .= data[1][:,1]
    θ0[N+1:2*N] .= data[2][:,2]
    θ0[nsims*N+1:end] .= 2*rand(xsize+N+1).-1

    # initialise vector of losses
    global loss_vec = [100.0]
    global iters = 0

    # build the optimization problem, lower and upper bounds for parameters are set here
    optprob = OptimizationProblem(optf, θ0, lb=-10*ones(length(θ0)),ub=10*ones(length(θ0)), callback=callback)

    # solve optimization problem using DE/1/rand/bin, a differential evolution algorithm. maximum iterations can be adjusted as needed
    result = Optimization.solve(optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 200000, callback=callback)
    θbest = result.u

    # get fitted ICs and model parameters from the result of the optimization problem
    ics = reshape(abs.(θbest[1:nsims*N]), (N, nsims))
    p = θbest[nsims*N+1:end]

    # make adjacency matrices for simple interpretation of results, and save all parameters to a .mat file for further processing
    (Aa,Ai) = make_matrices(p)
    mdic = Dict("pbest"=>θbest,"Aa"=>Aa,"Ai"=>Ai)
    matwrite("opts/opt_hill_gae_"*string(k)*".mat",mdic)
end