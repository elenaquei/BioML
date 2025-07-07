# BioML

In this project, we present an application of graph neural networks for constructing mechanistic models for time-dependent gene expression. Our result highlights not only the flexibility of GAEs, but also their limitations, appearing in the form of spurious edges. Furthermore, we demonstrate how a good fit of the data does not relate to a good representation of the underlying model, thus highlighting the importance of additional testing on fitted models, such as the presented knock-down experiments.

We have used BoolODE to generate synthetic data for this project (https://github.com/Murali-group/BoolODE). To use the code in this pipeline, the full output of BoolODE should be put into a subdirectory with the name of the simulated model in the data directory.

The github contains all files necessary to run the code and generate the figures in the manuscript. Specifically:

# In the main directory:

GAE_improving_network.ipynb: Includes the code needed to run our GAE on a given network, iteratively removing one edge from the network and outputting + saving the predicted adjacency matrices for these cases. This code is used to generate the GRN inference results presented in Figures 3 and 4 of the paper.

optimise_ODE.jl: Contains all code necessary to optimise the ODE model based on the improved graph (or any other desired graph structure) and given data. Output is saved to a .mat file. This code is used to generate Figure 6B (middle) in the paper.

kd_experiment.jl: Contains the code necessary to run the knock-down experiments, loading information from the .mat file. This code is used to generate Figure 6B (right) in the paper.

# In the visualisation directory:

fig3_get_statistics_gae_results.m: contains the code for generating the graph plots in Figure 3 of the manuscript provided output data from the GAE is available for all edges in the original graph.

fig3_parameter_tuning_visualisation.m: contains the code for the visualisation of the results for the hyperparameter sweep performed for the GAE.

fig4_ROC_plots.m: contains the code to plot the ROC curves in Figure 4 of the manuscript for a given network.

# In the utils directory:

boolODE_data_to_pyg_data.py: contains code to convert the output of BoolODE to data compatible with pytorch geometric.
