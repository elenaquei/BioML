# BioML

In this project, we present an application of graph neural networks for constructing mechanistic models for time-dependent gene expression. Our result highlights not only the flexibility of GAEs, but also their limitations, appearing in the form of spurious edges. Furthermore, we demonstrate how a good fit of the data does not relate to a good representation of the underlying model, thus highlighting the importance of additional testing on fitted models, such as the presented knock-down experiments.

We have used BoolODE to generate synthetic data for this project (https://github.com/Murali-group/BoolODE). To use the code in this pipeline, the full output of BoolODE should be put into a subdirectory with the name of the simulated model in the data directory.

The github contains all files necessary to run the code and generate the figures in the manuscript. Specifically:

# In the main directory:

GAE_improving_network.ipynb: Includes the code needed to run our GAE on a given network, iteratively removing one edge from the network and outputting + saving the predicted adjacency matrices for these cases. This code is used to generate the GRN inference results presented in Figures 3 and 4 of the paper.

optimise_ODE.jl: Contains all code necessary to optimise the ODE model based on the improved graph (or any other desired graph structure) and given data. Output is saved to a .mat file. This code is used to generate Figure 6B (middle) in the paper.

kd_experiment.jl: Contains the code necessary to run the knock-down experiments, loading information from the .mat file. This code is used to generate Figure 6B (right) in the paper.

# In the visualisation directory:

get_statistics_gae_results.m: contains all code necessary to get the graph plots presented in Figure 4 of the manuscript, along with the data to construct the bar graphs in figure 5 of the manuscript.

make_plots_for_comparing_graphs.m: contains the code necessary for running the analysis on multiple fits of the ODE model (including statistics discussed in Chapter 3).

plot_solution.jl: contains the code necessary to plot the results of the ODE model (used in Figure 6 of the manuscript).

plot2graphs.jl: contains the code to plot two graphs: the input to the GAE and the output of the GAE using red edges to indicate added edges.

# In the utils directory:

boolODE_data_to_pyg_data.py: contains code to convert the output of BoolODE to data compatible with pytorch geometric.

characterise_simulations.ipynb: contains all code for splitting up the simulation of the bifurcating graph into two distinct trajectories.

extract_adjacency.jl: contains code to convert the txt output files of plot_solution.jl to mat files used in the visualisation functions.