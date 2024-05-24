clear

for k=0:2
    load(['per_fit_restricted_n',num2str(k),'.mat']);
    ot = offdiag+symm;
    if min(ot) < 0.1
        disp(k)
        histogram(ot)
    end
end