%  In this script, ROC curves are plotted for the six synthetic networks, compared to three alternative GRN inference methods.
%  It assumes results for the three compared algorithms are in [model_name]/[algorithm_name]/outFile.txt
gae_pth = 'new_output\';

network = 'dyn_linear';

pattern = fullfile(gae_pth,[network, '*']);

files = dir(pattern);

% Step 2: Filter out those that have extra text before the first underscore after the prefix
if strcmp(network,'dyn_bifurcating') || strcmp(network, 'dyn_linear')
    filteredFiles = {};
    for k = 1:length(files)
        fname = files(k).name;
        % Use regexp to ensure the name matches pattern: dyn_bifurcating_<digits>.mat
        if ~isempty(regexp(fname, ['^',network,'_\d+_\d+\.mat$'], 'once'))
            filteredFiles{end+1} = fname;
        end
    end
    filenms = filteredFiles;
else
    filenms = {files.name};
end

% we load in the networks and add them to a list, to make a full overview of predicted edges
genie3vec = [];
pidcvec = [];
grnboost2vec = [];
gaevec = [];
refvec = [];

for k=1:length(filenms)
    genie3net = build_adjacency(network,'GENIE3');
    pidcnet = build_adjacency(network,'PIDC');
    grnboost2net = build_adjacency(network,'GRNBOOST2');
    
    refnet = build_refnetwork(network);

    gaenet = load(['new_output\',filenms{k}]);
    gaenet = gaenet.inferred_adj;

    refnet = refnet(:);
    gaenet = gaenet(:);
    genie3net = genie3net(:);
    pidcnet = pidcnet(:);
    grnboost2net = grnboost2net(:);

    refnet(isnan(gaenet))= [];
    genie3net(isnan(gaenet))= [];
    pidcnet(isnan(gaenet))= [];
    grnboost2net(isnan(gaenet))= [];
    gaenet(isnan(gaenet))= [];

    refvec = [refvec;refnet];
    gaevec = [gaevec;gaenet];
    genie3vec = [genie3vec;genie3net];
    pidcvec = [pidcvec;pidcnet];
    grnboost2vec = [grnboost2vec;grnboost2net];
end


% plot ROC curves
figure; hold on; grid on;

% GAE
[fpr, tpr, ~, auroc_gae] = perfcurve(refvec, gaevec, 1);
plot(fpr, tpr, 'LineWidth', 2);

% GENIE3
[fpr, tpr, ~, auroc_genie3] = perfcurve(refvec, genie3vec, 1);
plot(fpr, tpr, 'LineWidth', 2);

% PIDC
[fpr, tpr, ~, auroc_pidc] = perfcurve(refvec, pidcvec, 1);
plot(fpr, tpr, 'LineWidth', 2);

% GRNBOOST2
[fpr, tpr, ~, auroc_grnboost2] = perfcurve(refvec, grnboost2vec, 1);
plot(fpr, tpr, 'LineWidth', 2);

plot([0,1],[0,1],'k-.','LineWidth',2)

xlabel('False Positive Rate','FontSize',16);
ylabel('True Positive Rate','FontSize',16);

lgd = legend( ...
    sprintf('GAE (AUROC = %.4f)', auroc_gae), ...
    sprintf('GENIE3 (AUROC = %.4f)', auroc_genie3), ...
    sprintf('PIDC (AUROC = %.4f)', auroc_pidc), ...
    sprintf('GRNBOOST2 (AUROC = %.4f)', auroc_grnboost2), ...
    'Random predictor',...
    'Location', 'SouthEast');

lgd.FontSize = 16;
fig=gcf;
fig.Position=[100,100,600,600];

set(gca,'FontSize',16)
print(gcf,[network,'.png'],'-dpng','-r500')