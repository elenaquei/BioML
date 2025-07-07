# BioML

In this project, we present an application of graph neural networks for constructing mechanistic models for time-dependent gene expression. Our result highlights not only the flexibility of GAEs, but also their limitations, appearing in the form of spurious edges. Furthermore, we demonstrate how a good fit of the data does not relate to a good representation of the underlying model, thus highlighting the importance of additional testing on fitted models, such as the presented knock-down experiments.

We have used BoolODE to generate synthetic data for this project (https://github.com/Murali-group/BoolODE). To use the code in this pipeline, the full output of BoolODE should be put into a subdirectory with the name of the simulated model in the data directory.

The github contains all files necessary to run the code and generate the figures in the manuscript. Specifically:

# In the main directory:

GAE_improving_network.ipynb: Includes the code needed to run our GAE on a given network, iteratively removing one edge from the network and outputting + saving the predicted adjacency matrices for these cases. This code is used to generate the GRN inference results presented in Figures 3 and 4 of the paper.

optimise_ODE.jl: Contains all code necessary to optimise the ODE model based on the improved graph (or any other desired graph structure) and given data. Output is saved to a .mat file. This code is used to generate Figure 6B (middle) in the paper.

kd_experiment.jl: Contains the code necessary to run the knock-down experiments, loading information from the .mat file. This code is used to generate Figure 6B (right) in the paper.

# In the visualisation directory:

fig3_get_statistics_gae_results.m: contains the code for generating the graph plots in Figure 3 of the paper provided output data from the GAE is available for all edges in the original graph.

fig3_parameter_tuning_visualisation.m: contains the code for the visualisation of the results for the hyperparameter sweep performed for the GAE.

fig4_ROC_plots.m: contains the code to plot the ROC curves in Figure 4 of the paper for a given network.

fig4_cutoff_testing.m: contains the code for manually setting the cutoff at which an edge is added to the network, and computing the AUROC of the inferred network as a binary classifier for this cut-off value.

fig5_GAE_scRNA.ipynb: contains the code for running our GAE on real-world scRNA-seq data.

fig5_boxplots.m: contains the code for generating the boxplots used to show the performance of our GAE on real-world data (Figure 5 in the paper).

fig6_simulate_ODE.jl: contains the code to generate the phase portaits shown in Figure 6B (middle) of the paper.

build_adjacency.m: contains helper code to build an adjacency matrix from the output generated in the BEELINE package for GRN inference (https://github.com/Murali-group/Beeline, used for benchmarking).

# In the utils directory:

boolODE_data_to_pyg_data.py: contains code to convert the output of BoolODE to data compatible with pytorch geometric.

# Running the Julia code

All Julia code has been run on Julia v1.11 with the following dependencies:

[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DiffEqParamEstim = "1130ab10-4a5a-5621-a13d-e4788d82bd4c"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationBBO = "3e6eede4-6085-4f62-9a71-46d9bc1eb92b"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
OptimizationOptimisers = "42dfb2eb-d2b4-4451-abcd-913932933ac1"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
