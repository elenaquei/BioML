% helper function to build an adjacency matrix for the different network inference methods 
function [A] = build_adjacency(model_nm, alg_nm)
    edge_table = readtable([model_nm,'/',alg_nm,'/outFile.txt']);
    
    if strcmp(alg_nm,'PIDC')
        src_genes = edge_table.Var1;
        tgt_genes = edge_table.Var2;
        weights = edge_table.Var3;
    else
        src_genes = edge_table.TF;
        tgt_genes = edge_table.target;
        weights = edge_table.importance;
    end
    A = zeros(2,2);
    
    % fill in adjacency matrix
    for k=1:length(src_genes)
        src = src_genes{k};
        tgt = tgt_genes{k};

        src_ind = str2num(src(2:end));
        tgt_ind = str2num(tgt(2:end));
        
        A(src_ind,tgt_ind) = weights(k);
    end

    % pad with zeros if necessary
    sz = size(A);

    if sz(1) > sz(2)
        A = [A, zeros(sz(1),sz(1)-sz(2))];
    elseif sz(2) > sz(1)
        A = [A; zeros(sz(2)-sz(1),sz(2))];
    end
    
    rm = [];
    % remove unused nodes
    for k=1:length(A(1,:))
        if sum(A(k,:))+sum(A(:,k))==0
            rm = [rm, k];
        end
    end

    A(rm,:) = [];
    A(:,rm) = []; 
   
end

