gr1 = find(offdiag < 0.1);
gr2 = find(symm < 0.1);
gr = intersect(gr1,gr2);

gr_c = setdiff(1:2000,gr);

split_data =  zeros(50,1);

x_gr = [];
y_gr = [];

x_gr_c = [];
y_gr_c = [];

distvec = [];
avdist = [];
for k = 1:length(l)
    x1 = [X(k,1,1),X(k,1,2)];
    x2 = [X(k,2,1),X(k,2,2)];
    sep1 = x1(1) > x1(2);
    sep2 = x2(1) > x2(2);
    if sep1 ~= sep2
        split_data(k) = 1;
    end

    if ~isempty(find(gr==k))
        x_gr = [x_gr,x1(1),x2(1)];
        y_gr = [y_gr,x1(2),x2(2)];
    else
        x_gr_c = [x_gr_c,x1(1),x2(1)];
        y_gr_c = [y_gr_c,x1(2),x2(2)];
    end

    dist1 = abs(x1(2) - x1(1));
    dist2 = abs(x2(2) - x2(1));
    distvec = [distvec, min(dist1,dist2)];
    avdist = [avdist, mean([dist1,dist2])];
end

grvec = zeros(2000,1);
grvec(gr) = 1;
figure
plot(distvec,symm+offdiag,'o')
xlabel('Minimum distance to separatrix between datapoints (|x-y|)')
ylabel('|W_1_1|+|W_2_2| + |W_1_2-W_2_1|')

figure
plot(distvec,offdiag,'o')
xlabel('Minimum distance to separatrix between datapoints (|x-y|)')
ylabel('|W_1_1+W_2_2|')

figure
plot(distvec,symm,'o')
xlabel('Minimum distance to separatrix between datapoints (|x-y|)')
ylabel('|W_1_2-W_2_1|')

figure
plot(avdist,symm+offdiag,'o')
xlabel('Average distance to separatrix between datapoints (|x-y|)')
ylabel('|W_1_1|+|W_2_2| + |W_1_2-W_2_1|')

figure
plot(avdist,offdiag,'o')
xlabel('Average distance to separatrix between datapoints (|x-y|)')
ylabel('|W_1_1+W_2_2|')

figure
plot(avdist,symm,'o')
xlabel('Average distance to separatrix between datapoints (|x-y|)')
ylabel('|W_1_2-W_2_1|')

[dat1,edges1] = histcounts2(x_gr,y_gr,15);

[dat2,edges2] = histcounts2(x_gr_c,y_gr_c,15);

figure
imagesc(edges1(1),edges1(2),flip(dat1));
xticklabels({0    0.7143    1.4286    2.1429    2.8571    3.5714    4.2857    5.0000})
yticklabels({5.0000    4.2857    3.5714    2.8571    2.1429    1.4286    0.7143         0})
title('Position of data points for good fits')
figure
imagesc(edges2(1),edges2(2),flip(dat2));
xticklabels({0    0.7143    1.4286    2.1429    2.8571    3.5714    4.2857    5.0000})
yticklabels({5.0000    4.2857    3.5714    2.8571    2.1429    1.4286    0.7143         0})
title('Position of data points for bad fits')

gr_split = find(split_data);
gr_split_c = setdiff(1:2000,gr_split);

length(intersect(gr_split,gr))
length(intersect(gr,gr_split_c))
length(intersect(gr_c,gr_split))
length(intersect(gr_c,gr_split_c))

A = [length(intersect(gr_split,gr)),length(intersect(gr,gr_split_c));length(intersect(gr_c,gr_split)),length(intersect(gr_c,gr_split_c))];

[h,p,stats] = fishertest(A);