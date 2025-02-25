a = open("data/dyn_bifurcating/fit_cluster0.txt") do f
    readlines(f) |> (s->parse.(Float64, s))
end

data_name = "dyn_bifurcating"

edge_trained = []

improved = true

adj = matread(data_path*data_name*"/net.mat")["Aref"]

global kd = ones(size(adj,1))
global kd[2] = 10.0

if edge_trained != []
      if improved == true
          if data_name == "dyn_trifurcating_new"
              data_name = "dyn_trifurcating"
          end
          print("gae_results/"*data_name*string(edge_trained[1]-1)*"_"*string(edge_trained[2]-1)*".mat")
          adj = matread("gae_results/"*data_name*"_"*string(edge_trained[1]-1)*"_"*string(edge_trained[2]-1)*".mat")["inferred_adj"]
          adj = Float64.(isnan.(adj))+Float64.(adj.>0.5)
      else
          adj[edge_trained[1],edge_trained[2]] = 0
      end
  end

global indices = get_indices(adj, 0.8)

function distribute_parameters(params, indices, size)
      # Construct the matrix without mutation
      A = zeros(eltype(params), matsize)
  
      for (i, p) in zip(indices, params)
          A[i[1],i[2]] = p
      end
  
      return A
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
      du[:] = -kd.*abs.(gamma).*u + abs.(b1) + abs.(wout).*tanh.(transpose(A) * u + b2)
  end

prob1 = ODEProblem(f, u0, [0.0,1.0], a)
sol1 = solve(prob1)
u = reduce(hcat, sol1.u)

ind = 3
# Plotting the data
scatter(tdat, dat[ind, :], label="Data", markersize=6, linewidth=2, legend=:topright)

# Plotting the solution
plot!(sol1.t, u[ind, :], label="Simulation", linewidth=3, color=:black)

# Customize the plot for publication quality
plot!(xlabel="Pseudotime", ylabel="Gene Expression", title="Gene "* string(ind), 
      legendfontsize=12, tickfontsize=10, 
      guidefontsize=14, titlefontsize=16)

# Adjust plot size
plot!(size=(600, 400))  # Small size for publication, adjust as needed

# Optional: Save the plot as a high-quality image
# savefig("plot.png", dpi=300)