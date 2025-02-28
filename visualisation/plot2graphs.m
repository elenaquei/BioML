% code to plot output of GAE against original graph
load("gae_results\dyn_bifurcating_2_4.mat")

adj1 = double(isnan(inferred_adj));

adj2 = double(inferred_adj>0.9) + double(isnan(inferred_adj));

G1 = digraph(adj1);
G2 = digraph(adj2);

% Compute node positions using a consistent layout
fig1 = figure;
h1 = plot(G1, 'NodeColor', 'k', 'EdgeColor', 'k', 'Layout', 'force', 'LineWidth', 1.5, 'MarkerSize', 8);
nodePos = [h1.XData' h1.YData']; % Store positions
axis off; box off;

% Expand axis limits slightly to accommodate self-loops
xlim([min(nodePos(:,1)) - 1, max(nodePos(:,1)) + 1]); 
ylim([min(nodePos(:,2)) - 1, max(nodePos(:,2)) + 1]); 
set(gca, 'FontSize', 16); % Increase font size

% Save figure 1
exportgraphics(fig1, 'graph1.png', 'Resolution', 300);

% Find edges in G2 but not in G1
[E1_s, E1_t] = find(adj1); % Edges in G1
[E2_s, E2_t] = find(adj2); % Edges in G2

E1 = [E1_s, E1_t]; % Edge list for G1
E2 = [E2_s, E2_t]; % Edge list for G2

% Identify unique edges in G2
newEdges = setdiff(E2, E1, 'rows'); % Edges present in G2 but not in G1

% Define edge colors
numEdgesG2 = numel(E2_s);
edgeColors = repmat([0 0 0], numEdgesG2, 1); % Default: black edges

% Find indices of unique edges and set them to red
for i = 1:size(newEdges,1)
    idx = find(E2(:,1) == newEdges(i,1) & E2(:,2) == newEdges(i,2)); 
    edgeColors(idx, :) = [1 0 0]; % Red color
end

% Plot second graph with edge coloring
fig2 = figure;
h2 = plot(G2, 'XData', nodePos(:,1), 'YData', nodePos(:,2), 'NodeColor', 'k', 'EdgeColor', 'k', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
for i = 1:numEdgesG2
    highlight(h2, E2(i,1), E2(i,2), 'EdgeColor', edgeColors(i,:), 'LineWidth', 1.5);
end
hold off;

axis off; box off;
xlim([min(nodePos(:,1)) - 1, max(nodePos(:,1)) + 1]); 
ylim([min(nodePos(:,2)) - 1, max(nodePos(:,2)) + 1]); 
set(gca, 'FontSize', 16); % Increase font size

% Save figure 2
exportgraphics(fig2, 'graph2.png', 'Resolution', 300);