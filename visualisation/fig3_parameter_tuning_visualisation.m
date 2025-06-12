% this script assumes a variable grid is available with size (A,B,C) containing the AUROC values for a validation network.
% Moreover, three vectors lr, hdim and epochs are expected to be available.
results = [];

sz = size(grid);

% reshape data
for i = 1:sz(1)
    for j = 1:sz(2)
        for k = 1:sz(3)
            row = [double(lr(i)), double(hdim(j)), double(epochs(k)), double(grid(i,j,k))];
            results = [results; row];
        end
    end
end

% convert to table with column names
T = array2table(results, ...
    'VariableNames', {'Learning rate', 'Hidden dimension', 'Epochs', 'AUROC'});

% normalize AUROC to [0, 1] for setting colors
norm_auroc = (T.AUROC - min(T.AUROC)) / ...
            (max(T.AUROC) - min(T.AUROC));

% set colormap
cmap = autumn(256);
idx = round(norm_auroc * 255) + 1;
rgb_colors = cmap(idx, :);

% add group ID column to table (needed for individual coloring)
T.GroupID = categorical((1:height(T))');

% add RGB triplets to a table (1 row per experiment)
T.R = rgb_colors(:,1);
T.G = rgb_colors(:,2);
T.B = rgb_colors(:,3);

% plot visualisation of hyperparameter optimisation using matlab's parallelplot
p = parallelplot(T, ...
    'CoordinateVariables', {'Learning rate', 'Hidden dimension', 'Epochs', 'AUROC'}, ...
    'GroupVariable', 'GroupID', ...
    'Color', [T.R T.G T.B], ...
    'LineWidth', 1.5, 'FontSize', 16);