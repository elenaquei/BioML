% In this script, we vary the threshold of the cut-off used to add edges and assess the effect on the AUROC. The results are shown in Figure 3D.
% It assumes results for the three compared algorithms are in [model_name]/[algorithm_name]/outFile.txt
threshold = 0:0.01:1;
auroc_vec = zeros(length(threshold),1);

% repeat for each possible threshold
for j=1:length(threshold)

% set paths in which to find the output of the GAE
gae_pth = 'new_output\';

network = 'dyn_linear-long';

pattern = fullfile(gae_pth,[network, '*']);

files = dir(pattern);

% filter out those that have extra text before the first underscore after the prefix
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

    % cutoff gae at threshold
    gaenet = double(gaenet>threshold(j));

    refvec = [refvec;refnet];
    gaevec = [gaevec;gaenet];
    genie3vec = [genie3vec;genie3net];
    pidcvec = [pidcvec;pidcnet];
    grnboost2vec = [grnboost2vec;grnboost2net];
end

% GAE
[fpr, tpr, ~, auroc_gae] = perfcurve(refvec, gaevec, 1);

auroc_vec(j) = auroc_gae;
end

% plot results
plot(threshold,auroc_vec,'LineWidth',2)
xlabel('Edge selection cut-off')
ylabel('AUROC')
ylim([0,1])
fontsize(gcf,18,"points")