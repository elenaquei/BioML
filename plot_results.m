% Create directed graph
load('2_4_improved_adj.mat')

G = digraph(A);
figure
% Extract edge weights and signs
weights = abs(G.Edges.Weight);
colors = G.Edges.Weight > 0; % Logical array: 1 for positive, 0 for negative

% Define improved color scheme (publication-friendly)
positiveColor = [0.2 0.2 0.8]; % Dark blue for positive weights
negativeColor = [0.8 0.2 0.2]; % Dark red for negative weights

% Assign colors
edgeColors = zeros(length(colors), 3);
edgeColors(colors == 1, :) = repmat(positiveColor, sum(colors), 1);
edgeColors(colors == 0, :) = repmat(negativeColor, sum(~colors), 1);

% Plot graph
h = plot(G, 'Layout', 'force', 'EdgeCData', colors, 'LineWidth', weights);

% Apply custom styles
h.NodeColor = [0 0 0]; % Black nodes
h.MarkerSize = 8; % Larger nodes
h.NodeFontSize = 14; % Increase font size for readability
h.EdgeAlpha = 0.8; % Slight transparency for better visibility
h.ArrowSize = 15; % **Increase arrow size** for better visibility
set(h, 'EdgeCData', colors); % Apply color mapping

nodePos = [h.XData' h.YData']; % Store positions

% Use the custom colormap
colormap([negativeColor; positiveColor]);
box off
axis off
print(gcf,'improved_bif_2_4.png','-dpng','-r300')

load('2_4_imputed_adj.mat')

G = digraph(A);
figure
% Extract edge weights and signs
weights = abs(G.Edges.Weight);
colors = G.Edges.Weight > 0; % Logical array: 1 for positive, 0 for negative

% Define improved color scheme (publication-friendly)
positiveColor = [0.2 0.2 0.8]; % Dark blue for positive weights
negativeColor = [0.8 0.2 0.2]; % Dark red for negative weights

% Assign colors
edgeColors = zeros(length(colors), 3);
edgeColors(colors == 1, :) = repmat(positiveColor, sum(colors), 1);
edgeColors(colors == 0, :) = repmat(negativeColor, sum(~colors), 1);

% Plot graph
h = plot(G, 'XData', nodePos(:,1), 'YData', nodePos(:,2), 'EdgeCData', colors, 'LineWidth', weights);

% Apply custom styles
h.NodeColor = [0 0 0]; % Black nodes
h.MarkerSize = 8; % Larger nodes
h.NodeFontSize = 14; % Increase font size for readability
h.EdgeAlpha = 0.8; % Slight transparency for better visibility
h.ArrowSize = 15; % **Increase arrow size** for better visibility
set(h, 'EdgeCData', colors); % Apply color mapping

nodePos = [h.XData' h.YData']; % Store positions

% Use the custom colormap
colormap([negativeColor; positiveColor]);
box off
axis off

print(gcf,'imputed_bif_2_4.png','-dpng','-r300')

load('2_4_original_adj.mat')

G = digraph(A);
figure
% Extract edge weights and signs
weights = abs(G.Edges.Weight);
colors = G.Edges.Weight > 0; % Logical array: 1 for positive, 0 for negative

% Define improved color scheme (publication-friendly)
positiveColor = [0.2 0.2 0.8]; % Dark blue for positive weights
negativeColor = [0.8 0.2 0.2]; % Dark red for negative weights

% Assign colors
edgeColors = zeros(length(colors), 3);
edgeColors(colors == 1, :) = repmat(positiveColor, sum(colors), 1);
edgeColors(colors == 0, :) = repmat(negativeColor, sum(~colors), 1);

% Plot graph
h = plot(G, 'XData', nodePos(:,1), 'YData', nodePos(:,2), 'EdgeCData', colors, 'LineWidth', weights);

% Apply custom styles
h.NodeColor = [0 0 0]; % Black nodes
h.MarkerSize = 8; % Larger nodes
h.NodeFontSize = 14; % Increase font size for readability
h.EdgeAlpha = 0.8; % Slight transparency for better visibility
h.ArrowSize = 15; % **Increase arrow size** for better visibility
set(h, 'EdgeCData', colors); % Apply color mapping

nodePos = [h.XData' h.YData']; % Store positions

% Use the custom colormap
colormap([negativeColor; positiveColor]);
box off
axis off

print(gcf,'original_bif_2_4.png','-dpng','-r300')

load('2_4_alt_original_adj.mat')

G = digraph(A);
figure
% Extract edge weights and signs
weights = abs(G.Edges.Weight);
colors = G.Edges.Weight > 0; % Logical array: 1 for positive, 0 for negative

% Define improved color scheme (publication-friendly)
positiveColor = [0.2 0.2 0.8]; % Dark blue for positive weights
negativeColor = [0.8 0.2 0.2]; % Dark red for negative weights

% Assign colors
edgeColors = zeros(length(colors), 3);
edgeColors(colors == 1, :) = repmat(positiveColor, sum(colors), 1);
edgeColors(colors == 0, :) = repmat(negativeColor, sum(~colors), 1);

% Plot graph
h = plot(G, 'XData', nodePos(:,1), 'YData', nodePos(:,2), 'EdgeCData', colors, 'LineWidth', weights);

% Apply custom styles
h.NodeColor = [0 0 0]; % Black nodes
h.MarkerSize = 8; % Larger nodes
h.NodeFontSize = 14; % Increase font size for readability
h.EdgeAlpha = 0.8; % Slight transparency for better visibility
h.ArrowSize = 15; % **Increase arrow size** for better visibility
set(h, 'EdgeCData', colors); % Apply color mapping

nodePos = [h.XData' h.YData']; % Store positions

% Use the custom colormap
colormap([negativeColor; positiveColor]);
box off
axis off

