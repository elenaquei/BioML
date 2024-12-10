using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationMOI, OptimizationOptimisers, JLD2, MAT

# function definition for fitted model
function f(du, u, p, t)
    W1 = [[p[1],p[2]] [p[3],p[4]]]
    W2 = [[p[5],p[6]] [p[7],p[8]]]
    b1 = [p[9],p[10]]
    b2 = [p[11],p[12]]
    gamma = [p[13],p[14]]
    du .= gamma.*u + b1 + W2*tanh.(W1*u+b2)
end

# use log thresholding to find adjacency matrix resulting from optimization
function find_network(W)
    Wsort = sort(log.(abs.(vcat(W...))))
    threshold = Wsort[findmax(Wsort[2:end]-Wsort[1:end-1])[2]+1]
    Aa = Float32.(W.>=threshold)
    Ai = Float32.(W.<=-threshold)
    A = Aa-Ai
    return A
end

function regularization(p, tgt, lambda)
    W1 = [p[1:2] p[3:4]]
    W2 = [p[5:6] p[7:8]]

    tgt = [tgt[1:2] tgt[3:4]]

    Ahat = W2*W1

    reg_loss = 0

    for i=1:size(Ahat)[1]
        for j=1:size(Ahat)[2]
            if tgt[i,j] == 0
                reg_loss = reg_loss + abs(Ahat[i,j])
            end
        end
    end

    return lambda*reg_loss
end

# run optimization for a specific ODE problem, initial conditions and a given optimizer
function run_optim(data, initial_conditions, optimizer, data_times, reg, iter)

    N = convert(Int32, size(initial_conditions)[1])

    # Building a loss function
    losses = [L2Loss(data_times, data[:, :, i]) for i in 1:N]
    loss(sim) = sum(losses[i](sim[i]) for i in 1:N)

    # now set up the problem with new parameters (randomly sampled)
    prob = ODEProblem(f, [1.0, 1.0], (0.0, 1.0), 2 .*rand(14,1).-1)

    function prob_func(prob, i, repeat)
        ODEProblem(prob.f, initial_conditions[i], prob.tspan, prob.p)
    end
    enprob = EnsembleProblem(prob, prob_func = prob_func)
    sim = solve(enprob, Tsit5(), trajectories = N, saveat = data_times)

    # build objective function
    obj = build_loss_objective(enprob, Tsit5(), loss, Optimization.AutoForwardDiff(), reg,
                            trajectories = N,
                            saveat = data_times)

    # solve optimization problem
    optprob = OptimizationProblem(obj, vec(2 .*rand(14,1).-1))

    result = solve(optprob, optimizer, maxiters = 20000)

    param = result.u

    # find network corresponding to output
    W1 = [param[1:2] param[3:4]]
    W2 = [param[5:6] param[7:8]]

    A = find_network(W2*W1)

    # save optimization results
    dat = Dict("p"=>param,"data"=>data,"f"=>f,"dimODE"=>length(initial_conditions[1]),"dimparam"=>length(param),"adj"=>A,"data_times"=>data_times)
    save("ADAM_results/fit_no_"*string(iter)*".jld2", "dat", dat)
    return param, A
end

# run n_optims amount of optimizations for ODE problem prob, with specific optimizer and a vector Nvec indicating how many ICs should be sampled
function run_optims(prob,optimizer,n_optims,tgt_name)

    correct = 0

    # run n_optims optimizations for this value of N
    for j in 1:n_optims
        print(j)

        dat = matread("network" * string(j)*".mat")

        p_init = dat["p"]
        
        x = dat["x"]

        N = convert(Int32, size(x)[2]/2)

        data_times = [0.0,1.0]

        W1 = [p_init[1:2] p_init[3:4]]
        W2 = [p_init[5:6] p_init[7:8]]

        A_truth = find_network(W2*W1)

        tgt = dat[tgt_name]

        reg = (p) -> regularization(p, tgt, 0.001)

        # Sample N initial conditions on the interval [0,5]
        initial_conditions = Vector{Vector{Float64}}(undef,N)
        for k in 1:N
            initial_conditions[k] = x[:,k]
        end

        # set up ensemble ODE problem solving for the different initial conditions
        function prob_func(prob, i, repeat)
            ODEProblem(prob.f, initial_conditions[i], prob.tspan, prob.p)
        end

        enprob = EnsembleProblem(prob, prob_func = prob_func)

        # generate data from sampled ICs (saved on time points data_times)
        sim = solve(enprob, Tsit5(), trajectories = N, saveat = data_times)
        data = Array(sim)

        # run optimization finding both full parameters and the resulting inferred adjacency matrix
        param, A = run_optim(data, initial_conditions,optimizer,data_times, reg, j)

        if sum(abs.(A - A_truth)) < 0.001
            correct = correct + 1
        end
    end

    return correct
end

# initial definition of ODE model
u0 = [0.3; 1.0]
tspan = (0.0, 1.0)
p = [0,-1,-1,0,2,0,0,2,2,2,2,2,-1,-1]
prob = ODEProblem(f, u0, tspan, p)

opt = Optimisers.Adam(0.01)

Noptims = 100

# tgt_name = "Ainit"
tgt_name = "Ahat"

# get fraction of correct optimizations as output
correct = run_optims(prob,opt,Noptims,tgt_name)/Noptims
