eps = 0.0:0.05:1;

string(eps(end))


res_vec = zeros(length(eps),1);
other_elems = zeros(length(eps),1);

for j=1:length(eps)
    disp(eps(j))
    if eps(j) == 0
        load('param_fit_eps_0.0.mat');
    elseif (eps(j) > 0.3 && eps(j) < 0.31) || (eps(j) > 0.59 && eps(j) < 0.61) || (eps(j) > 0.69 && eps(j) < 0.71)
        load(['param_fit_eps_',num2str(eps(j)),'0.mat']);
    else
        disp('check')
        disp(eps(j))
        load(['param_fit_eps_',num2str(eps(j)),'.mat']);
    end

    inc = 0;
    other_elem_cur = 0;
    p = (1-eps(j));
    for k=1:length(W1)
        scale = [1,(1-p),(1-p),p];
        vec = [W1(k,2,1),W1(k,3,2),W1(k,1,3),W1(k,1,2)];
        
        inc = inc + (sum(scale.*double(vec<0))/sum(scale));
        other_elem_cur = other_elem_cur + sum(sum(abs(W1(k,:,:)))) - sum(abs(vec));
    end
    
    other_elems(j) = other_elem_cur/(length(W1));
    res_vec(j) = inc/(length(W1));
end
figure
plot(eps,flipud(res_vec));
figure 
plot(eps,flipud(other_elems));