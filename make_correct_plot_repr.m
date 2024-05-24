
restricted = [];
for k=2:10
    load(['restr_repr_n',num2str(k),'.mat']);
    [p_av,p_cnt,p_min,p_split,correct,rho_av,rho_cnt,rho_min,A] = evaluate_fits_repr(X,l,symm,offdiag,W1,0);
    restricted = [restricted, correct];
end

h_restr= restricted + 1.96*sqrt(restricted.*(1-restricted)/100);
l_restr = restricted - 1.96*sqrt(restricted.*(1-restricted)/100);
e_restr = [l_restr,fliplr(h_restr)];

inside = [];
for k=2:10
    load(['repr_inside_n',num2str(k),'.mat']);
    [p_av,p_cnt,p_min,p_split,correct,rho_av,rho_cnt,rho_min,A] = evaluate_fits_repr(X,l,symm,offdiag,W1,0);
    inside = [inside, correct];
end

h_inside = inside + 1.96*sqrt(inside.*(1-inside)/100);
l_inside = inside - 1.96*sqrt(inside.*(1-inside)/100);
e_inside = [l_inside,fliplr(h_inside)];

% mask11 = [];
% for k=2:10
%     load(['mask11_n',num2str(k),'.mat']);
%     [p_av,p_cnt,p_min,p_split,correct,rho_av,rho_cnt,rho_min,A] = evaluate_fits(X,l,symm,offdiag,W1,0);
%     mask11 = [mask11, correct];
% end
% 
% h_mask11 = mask11 + 1.96*sqrt(mask11.*(1-mask11)/100);
% l_mask11 = mask11 - 1.96*sqrt(mask11.*(1-mask11)/100);
% e_mask11 = [l_mask11,fliplr(h_mask11)];

mask11_22 = [];
for k=2:18
    load(['repr_mask_n',num2str(k),'.mat']);
    [p_av,p_cnt,p_min,p_split,correct,rho_av,rho_cnt,rho_min,A] = evaluate_fits_repr(X,l,symm,offdiag,W1,0);
    mask11_22 = [mask11_22, correct];
end

h_mask11_22 = mask11_22 + 1.96*sqrt(mask11_22.*(1-mask11_22)/100);
l_mask11_22 = mask11_22 - 1.96*sqrt(mask11_22.*(1-mask11_22)/100);
e_mask11_22 = [l_mask11_22,fliplr(h_mask11_22)];

%%
fig = figure();

x = 2:10;
plot(x,restricted,LineWidth=2)
hold on
plot(x,inside,LineWidth=2);
hold on
plot(2:18,mask11_22,LineWidth=2);
hold on
% x2 = [x,fliplr(x)];
% fill(x2,e_inside,'r','FaceAlpha',0.1)
% hold on
% fill(x2,e_mask11,'y','FaceAlpha',0.1)
% hold on
% fill(x2,e_mask11_22,'m','FaceAlpha',0.1)
% hold on
% fill(x2,e_restr,'b','FaceAlpha',0.1)
legend('Restricted','Inside: No mask','Inside: Mask all undesired elements')
ylabel('% of fits with correct relation x-|y-|z-|x')
xlabel('Amount of data points')
ylim([0,1]);
fontsize(fig, 14, "points")
