using DifferentialEquations, Plots, SciMLSensitivity, Zygote, Flux

function get_data(plt)
    x = zeros(2,Ndata)
    y = zeros(2,Ndata)
    
    for k in 1:Ndata
        u0 = 5*rand(2)
    
        prob = ODEProblem(tanh_ode,u0,[0.0f0,T],p)
    
        sol = solve(prob)
    
        u = reduce(hcat,sol.u)
        
        if plt
            if k == 1
                pl = plot(u[1,:],u[2,:])
            else
                pl = plot!(u[1,:],u[2,:])
                if k == Ndata
                    display(pl)
                end
            end
        end
        
        x[:,k] = u0
        y[:,k] = u[:,end]
    end

    return x,y
end

function tanh_ode(du,u,p,t)

    # set up parameters in a naive way
    W1 = [p[1] p[2];p[3] p[4]]
    #W2 = [p[5] p[6];p[7] p[8]]
    #W2 = [2 0;0 2]
    #b1 = [p[9]; p[10]]
    #b2 = [p[11]; p[12]]
    #gamma = [p[13]; p[14]]
    W2 = [2 0;0 2]
    b1 = [2; 2]
    b2 = [2; 2]
    gamma = [1; 1]

    # ODE definition
    du[:] = W2*tanh.(W1*u+b1)+b2-gamma.*u
end

function loss()
    lss = 0
    for k in 1:length(x[1,:])
        u0 = vec(x[:,k])
        prob = ODEProblem(tanh_ode,u0,[0.0f0,T],p)
        sol = solve(prob)
        u = reduce(hcat,sol.u)
        lss = lss + sum(abs2,u[:,end]-y[:,k])
    end

    return lss + 0.0001*sum(abs.(p))
end

cb = function()
    L = loss()
    print(L)
    print('\n')
    append!(lossvec, L)
    return false
end

global Ndata = 20
global T = 0.5

global Niter = 500
lr = 0.1

# p = vec(2*rand(4).-1)
# open("orig_p.txt","a") do io
#     println(io,"p=",p)
#  end
p = [0,-1,-1,0,2,0,0,2,2,2,2,2,1,1]

(x,y) = get_data(true)

p = vec(rand(4,1))
params = Flux.params(p)
iter = Iterators.repeated((), Niter)
opt = ADAM(lr)

# initialise loss vector
lossvec = zeros(0)

cb()
Flux.train!(loss,params,iter,opt,cb=cb)



