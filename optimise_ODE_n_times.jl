using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimisers, SparseArrays, MAT, CSV, DataFrames, Statistics, OptimizationBBO

global iter_counter = 0

data_name = "dyn_bifurcating"

edge_trained = [3,5]

improved = false

# cluster used for fitting bifurcating network
cluster = 0

function get_indices(A, threshold)
    indices = []
    print(typeof(adj))
    print(size(A))
    for i in 1:size(A)[1]
        for j in 1:size(A)[2]
            if abs(A[i,j]) > threshold
                push!(indices, (i,j))
            end
        end
    end
    return indices
end

data_path = "data/"

dat = Matrix(CSV.read(data_path * data_name * "/ExpressionData.csv", DataFrame, delim=","))[:,2:end]

tdat = Matrix(CSV.read(data_path * data_name * "/PseudoTime.csv", DataFrame, delim=","))[:,2]

if data_name == "dyn_bifurcating"
    cl = Matrix(CSV.read("clusters.csv", DataFrame, delim=","))
    cl = vec([0; cl])

    dat0 = Float64.(dat[:,cl.==0])
    dat1 = Float64.(dat[:,cl.==1])
    t0 = Float64.(tdat[cl.==0])
    t1 = Float64.(tdat[cl.==1])

    sorted_indices = sortperm(t0)
    t0= Float64.(t0[sorted_indices])
    dat0 = Float64.(dat0[:, sorted_indices])

    sorted_indices = sortperm(t1)
    t1= Float64.(t1[sorted_indices])
    dat1 = Float64.(dat1[:, sorted_indices])
     
    u0_0 = mean(dat0[:,t0.<0.05], dims=2)
    u0_1 = mean(dat1[:,t1.<0.05], dims=2)

    if cluster == 0
        tdat = t0
        dat = dat0
        u0 = u0_0
    else
        tdat = t1
        dat = dat1
        u0 = u0_1
    end
else
    tdat = Float64.(tdat)
    dat = Float64.(dat)
    u0 = mean(dat[:,tdat.<0.05], dims=2)
end

adj = matread(data_path*data_name*"/net.mat")["Aref"]

if edge_trained != []
    if improved == true
        if data_name == "dyn_trifurcating_new"
            data_name = "dyn_trifurcating"
        end
        print("gae_results/"*data_name*string(edge_trained[1]-1)*"_"*string(edge_trained[2]-1)*".mat")
        adj = matread("gae_results/"*data_name*"_"*string(edge_trained[1]-1)*"_"*string(edge_trained[2]-1)*".mat")["inferred_adj"]
        adj = Float64.(isnan.(adj))+Float64.(adj.>0.9)
    else
        adj[edge_trained[1],edge_trained[2]] = 0
    end
end

global indices = get_indices(adj, 0.8)

global matsize = (size(adj)[1], size(adj)[2])

N = 3

function distribute_parameters(params, indices, size)
    # Construct the matrix without mutation
    A = zeros(eltype(params), matsize)

    for (i, p) in zip(indices, params)
        A[i[1],i[2]] = p
    end

    return A
end

function reg(p)
    reg_loss = 5*sum(abs.(p))
    # println(typeof(p))
    reg_loss = reg_loss + 10000*sum(abs.(f(zeros(eltype(p),7),dat[:,end],p,0)))
    return reg_loss
end

function f(du, u, p, t)

    # distribute parameters into A, gamma, b1, b2, wout used in different parts of the model equations
    adj_p = p[1:length(indices)]
    gamma = p[length(indices)+1:length(indices)+matsize[1]]
    b1 = p[length(indices)+matsize[1]+1:length(indices)+matsize[1]*2]
    b2 = p[length(indices)+matsize[1]*2+1:length(indices)+matsize[1]*3]
    wout = p[length(indices)+matsize[1]*3+1:end]

    # distribute parameters according to given adjacency matrix
    A = distribute_parameters(adj_p, indices, matsize)

    # calculate right-hand side of ODE
    du[:] = -abs.(gamma).*u + abs.(b1) + abs.(wout).*tanh.(transpose(A) * u + b2)
end

function standardize_p(p)
    adj_p = p[1:length(indices)]
    gamma = p[length(indices)+1:length(indices)+matsize[1]]
    b1 = p[length(indices)+matsize[1]+1:length(indices)+matsize[1]*2]
    b2 = p[length(indices)+matsize[1]*2+1:length(indices)+matsize[1]*3]
    wout = p[length(indices)+matsize[1]*3+1:end]

    gamma = abs.(gamma)
    b1 = abs.(b1)
    wout = abs.(wout)

    return [adj_p; gamma; b1; b2; wout]
end

# callback function to monitor training
callback = function (state, l; doplot = false) 
    global iter_counter += 1
   
    # save loss to loss_vec
    push!(loss_vec,l)

    if l < min_loss
        global min_loss = l
        println("Iteration: $iter_counter")
        println("Current loss: ", l)
    end
    return false
end

# time span
tspan = (0.0, 1.0)
p = rand(length(indices) + 4*size(adj)[1])
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5())

cost_function = build_loss_objective(prob, Tsit5(), L2Loss(tdat,dat),
                                     Optimization.AutoForwardDiff(), reg,
                                     verbose = false, trajectories=N)
                                    
for n in 1:25
    global iter_counter = 0
    global min_loss = 100000
    pinit = rand(length(indices) + 4*size(adj)[1])
    global loss_vec = [cost_function(pinit)]

    lb = -10*ones(length(pinit))
    ub = 10*ones(length(pinit))

    optprob = Optimization.OptimizationProblem(cost_function, pinit, lb=lb, ub=ub)
    # optsol = solve(optprob, OptimizationOptimisers.Adam(0.2), callback=callback)

    @time optsol = solve(optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 100000,
    maxtime = 600.0, callback=callback)

    bestp = optsol.u
    bestp = standardize_p(bestp)
    println("Best parameters: ", bestp)

    # sorting naming for saving best parameters
    if cluster >= 0
        if edge_trained !=[]
            if improved == true
                filenm = data_path * data_name * "/fit_cluster"*string(cluster)*"_"*string(edge_trained[1]-1)*"_"*string(edge_trained[2]-1)*"_improved"*string(n)*".txt"
            else
                filenm = data_path * data_name * "/fit_cluster"*string(cluster)*"_"*string(edge_trained[1]-1)*"_"*string(edge_trained[2]-1)*"_"*string(n)*".txt"
            end
        else
            filenm = data_path * data_name * "/fit_cluster" * string(cluster) * "_"*string(n)*".txt"
        end
    else
        if edge_trained !=[]
            if improved == true
                filenm = data_path * data_name * "/fit_" * string(edge_trained[1]-1) * "_"*string(edge_trained[2]-1) * "_improved"*string(n)*".txt"
            else
                filenm = data_path * data_name * "/fit_" * string(edge_trained[1]-1) * "_"*string(edge_trained[2]-1) * "_"*string(n)*".txt"
            end
        else
            filenm = data_path * data_name * "/fit_"*string(n)*".txt"
        end
    end

    io = open(filenm, "w") do io
        for x in bestp
            println(io, x)
        end
    end
end