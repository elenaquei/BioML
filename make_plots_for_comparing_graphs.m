load('data\dyn_bifurcating\signed_net');
correct = 0;
incorrect = 0;
for k=1:20
    % Create directed graph
    load(['data\dyn_bifurcating\fit_cluster0_2_4_improved',num2str(k),'.mat'])
    A = Aref;
    
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

    colormap([negativeColor; positiveColor]);

    for i = 1:length(A(1,:))
        for j=1:length(A(1,:))
            if true_net(i,j) ~= 0 && A(i,j) ~=0
                if sign(A(i,j)) == sign(true_net(i,j))
                    correct = correct+1;
                else
                    incorrect = incorrect+1;
                end
            end
        end
    end
end