network_name = 'dyn_bifurcating';

if size(network_name) == size('dyn_trifurcating_new')
    if network_name == 'dyn_trifurcating_new'
        dataname = 'dyn_trifurcating';
    else
        dataname = network_name;
    end
else
    dataname = network_name;
end

if size(network_name) == size('dyn_linear-long')
    if network_name == 'dyn_linear-long'
        load('gae_results/dyn_linear-long_1_2.mat');
        Aref = isnan(inferred_adj);
        Aref(2,3) = 1;
    else
        Aref = read_ref_network(dataname);
    end
elseif size(network_name) == size('dyn_bifurcating_converging')
    if network_name == 'dyn_bifurcating_converging'
        load('gae_results/dyn_bifurcating_converging_0_2.mat');
        Aref = isnan(inferred_adj);
        Aref(1,3) = 1;
    else
        Aref = read_ref_network(dataname);
    end
else
    Aref = read_ref_network(dataname);
end

save(['data/',dataname,'/net.mat'],"Aref");

[Amean, Apred, Ainvpred] = get_statistics(Aref,network_name);

G2 = digraph(Apred);
G1 = digraph(double(Amean>0.5));

% w = 0.1 + 5*(G2.Edges.Weight-min(G2.Edges.Weight))/(max(G2.Edges.Weight)-min(G2.Edges.Weight));

tot = length(Apred(1,:))^2-sum(sum(Aref));

w = 0.1 + 5*(tot-G2.Edges.Weight)/tot;

figure
A1 = adjacency(G1);
A2 = adjacency(G2);
% Extract edges
[s1, t1] = find(A1); % Edges from G1
[s2, t2] = find(A2); % Edges from G2

% Combine edges into a single set
s = [s1; s2];
t = [t1; t2];

% Create a new graph that includes all edges
G = digraph(s, t);

% Plot the graph
h = plot(G, 'NodeFontSize',14,'NodeColor','k','MarkerSize',10, 'Layout', 'force', 'EdgeAlpha',0.4);

% Highlight edges
hold on
% Loop through edges to customize appearance
for i = 1:numel(s)
    if ismember([s(i), t(i)], [s1, t1], 'rows')  % Edge belongs to G1
        highlight(h, s(i), t(i), 'EdgeColor', 'r', 'LineWidth', 1.5);
    else  % Edge belongs to G2
        weightIndex = find(ismember([s2, t2], [s(i), t(i)], 'rows'));
        highlight(h, s(i), t(i), 'EdgeColor', 'k', 'LineWidth', w(weightIndex));
    end
end
box off
axis off
hold off

print([network_name,'.png'],'-dpng','-r300');

function [Amean, Apred, Ainvpred] = get_statistics(Aref, network_name)
    N = length(Aref(1,:));

    Amean = zeros(N,N);
    Apred = zeros(N,N);
    Ainvpred = zeros(N,N);
    
    nedges = 0;
    for k=1:length(Aref(1,:))
        for j=1:length(Aref(:,2))
            if Aref(k,j) ~= 0
                % load network per GAE training
                load(['gae_results/',network_name,'_',num2str(k-1),'_',num2str(j-1),'.mat'])
                inferred_adj(isnan(inferred_adj)) = 0;

                % get rank of inferred connection
                [Apred(k,j),Ainvpred(k,j)] = get_inv_rank(inferred_adj,k,j);

                % update mean inferred network
                inferred_adj(k,j) = 0;
                Amean = Amean + inferred_adj;
                nedges = nedges + 1;
            end
        end
    end

    Amean = Amean/nedges;
end

function [rank, inv_rank] = get_inv_rank(A,k,j)
    N = length(A(1,:));
    Avec = reshape(A,[N^2,1]);
    [~,ind] = maxk(Avec,N^2);
    f_ind = sub2ind([N,N],k,j);
    rank = find(ind==f_ind);
    inv_rank = 1/(rank/N^2);
end

function [Aref] = read_ref_network(name)
    M = readtable(['data/',name,'/refNetwork.csv']);
    M = M{:,["Gene1","Gene2","Type"]};
    Aref = zeros(2,2);

    % fill in adjacency matrix
    for k = 1:length(M)
        i1 = str2double(M{k,1}(2:end));
        i2 = str2double(M{k,2}(2:end));
        Aref(i1,i2) = 1;
    end
    
    % ensure that resulting matrix is square (regardless of which nodes are
    % getting incoming/outgoing connections)
    sz = size(Aref);
    if sz(1) > sz(2)
        diff = sz(1)-sz(2);
        Aref = [Aref, zeros(sz(1),diff)];
    elseif sz(2) > sz(1)
        diff = sz(2)-sz(1);
        Aref = [Aref; zeros(diff,sz(2))];
    end

    % remove any unused nodes
    rem = [];
    for k=1:length(Aref(1,:))
        if max(Aref(k,:)) == 0 && max(Aref(:,k)) == 0
            rem = [rem, k];
        end
    end
    Aref(rem,:) = [];
    Aref(:,rem) = [];
end