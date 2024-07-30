function [p_av,p_cnt,p_min,p_split,correct,rho_av,rho_cnt,rho_min,A] = evaluate_fits_repr(X,l,symm,offdiag,W1,plt,alt)
close all

if nargin < 7
    alt = 0;
end

sz = size(X);
N = sz(1); % find number of fits
Nin = sz(2); % find number of data points
Nvar = sz(3); % find number of state variables

if alt
    symm = [];
    for k=1:N
        W = [W1(k,1,1),W1(k,1,2);W1(k,2,1),W1(k,2,2)];
        symm = [symm,abs(W(1,2)+1.5)+abs(W(2,1)+0.8)];
    end
end

% find good fits and bad fits
[gr,gr_c] = find_good_fits(offdiag,symm,N);

% initialise tracked variables
x_gr = [];
y_gr = [];
x_gr_c = [];
y_gr_c = [];
mindist = [];
avdist = [];
split_dat =  zeros(N,1);
cnt_dat = zeros(N,1);

% loop over optimisations to find some metrics
correct = 0;
for k=1:N
    
    % find the initial conditions for the given optimisation set
    x = cell(Nin,1);
    for n=1:Nin
        for j=1:Nvar
            x{n} = [x{n},X(k,n,j)];
        end
    end
    
    % find whether there are initial conditions on each side of the
    % separatrix
    fnd = [0,0];
    cnt = [0,0];
    for n=1:Nin
        cond = x{n}(1)>x{n}(2);
        if cond == 0
            fnd(1)=1;
            cnt(1)=cnt(1)+1;
        else
            fnd(2)=1;
            cnt(2)=cnt(2)+1;
        end
    end

    split_dat(k) = 0;
    if sum(fnd) == 2
        split_dat(k) = 1;
    end

    cnt_dat(k) = abs(0.5-cnt(1)/(cnt(1)+cnt(2)));

    % organise initial into groups for plotting purposes (with good/bad
    % outcomes)
    for n = 1:Nin
        if ~isempty(find(gr==k))
            x_gr = [x_gr,x{n}(1)];
            y_gr = [y_gr,x{n}(2)];
        else
            x_gr_c = [x_gr_c,x{n}(1)];
            y_gr_c = [y_gr_c,x{n}(2)];
        end
    end
    
    % find distance to separatrix (only relevant for toggle-switch)
    dist = [];
    for n = 1:Nin
        dist = [dist, abs(x{n}(1)-x{n}(2))];
    end
    mindist = [mindist, min(dist)];
    avdist = [avdist, mean(dist)];
    
    % check if correct relation between x,y,z is found
    W = [W1(k,1,1),W1(k,1,2),W1(k,1,3);W1(k,2,1),W1(k,2,2),W1(k,2,3);W1(k,3,1),W1(k,3,2),W1(k,3,3)];
    W2 = abs(W);
    W2(2,1) = 0;
    W2(3,2) = 0;
    W2(1,3) = 0;
    
    correct = correct + double(W(2,1)<-0.5 && W(3,2)<-0.5 && W(1,3) < -0.5 && max(max(W2))<0.5);
end

correct = correct/N;

% compute statistics

% use Spearman's rank correlation to find if minimum distance/average
% distance is related to goodness of fit
[rho_min,p_min] = corr(mindist',(symm+offdiag)','Type','Spearman');
[rho_av,p_av] = corr(avdist',(symm+offdiag)','Type','Spearman');
[rho_cnt,p_cnt] = corr(cnt_dat,(symm+offdiag)','Type','Spearman');

% use Fisher's exact test to test relation between having points on either
% side of separatrix and goodness of fit
gr_split = find(split_dat);
gr_split_c = setdiff(1:N,gr_split);

length(intersect(gr_split,gr))
length(intersect(gr,gr_split_c))
length(intersect(gr_c,gr_split))
length(intersect(gr_c,gr_split_c))

A = [length(intersect(gr_split,gr)),length(intersect(gr,gr_split_c));length(intersect(gr_c,gr_split)),length(intersect(gr_c,gr_split_c))];

[~,p_split,~] = fishertest(A);

% plot results
if plt
    [dat1,edges1] = histcounts2(x_gr,y_gr,15);
    figure
    imagesc(edges1(1),edges1(2),flip(dat1));
    xticklabels({0    0.7143    1.4286    2.1429    2.8571    3.5714    4.2857    5.0000})
    yticklabels({5.0000    4.2857    3.5714    2.8571    2.1429    1.4286    0.7143         0})
    title('Position of data points for good fits')
    
    [dat2,edges2] = histcounts2(x_gr_c,y_gr_c,15);
    figure
    imagesc(edges2(1),edges2(2),flip(dat2));
    xticklabels({0    0.7143    1.4286    2.1429    2.8571    3.5714    4.2857    5.0000})
    yticklabels({5.0000    4.2857    3.5714    2.8571    2.1429    1.4286    0.7143         0})
    title('Position of data points for bad fits')
    
    figure
    plot(mindist,symm+offdiag,'o')
    xlabel('Minimum distance to separatrix between datapoints (|x-y|)')
    ylabel('|W_1_1|+|W_2_2| + |W_1_2-W_2_1|')
    
    figure
    plot(avdist,symm+offdiag,'o')
    xlabel('Average distance to separatrix between datapoints (|x-y|)')
    ylabel('|W_1_1|+|W_2_2| + |W_1_2-W_2_1|')
end
end


% function to find the fits that went well (i.e. low diagonal elements and
% almost symmetric offdiagonal elements)
function [gr,gr_c] = find_good_fits(offdiag,symm,N)
    
    % find fits that went well
    gr1 = find(offdiag < 0.1);
    gr2 = find(symm < 0.1);
    gr = intersect(gr1,gr2);

    % find fits that did not go well
    gr_c = setdiff(1:N,gr);
end