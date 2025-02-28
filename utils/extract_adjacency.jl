using MAT

base_file_nm = "data/dyn_bifurcating/no_b_fit_cluster0_"

data_name = "dyn_bifurcating"

edge_trained = []

improved = false

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

function distribute_parameters(params, indices, size)
      # Construct the matrix without mutation
      A = zeros(eltype(params), matsize)
  
      for (i, p) in zip(indices, params)
          A[i[1],i[2]] = p
      end
  
      return A
end

for i in 1:20
    a = open("$base_file_nm$i.txt") do f
        readlines(f) |> (s->parse.(Float64, s))
    end

    adj = distribute_parameters(a[1:length(indices)], indices, matsize)
    matwrite("$base_file_nm$i.mat", Dict("Aref" => adj))
end