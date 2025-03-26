function S_lb = lower_bound(par,d)
    % `lower_bound` determines the lower bound of bipartite quantum steerability
    % Input:
    %   par - visibility parameter of the target assemblage
    %   d - dimension of the system
    % Output:
    %   S_lb - lower bound of bipartite quantum steerability
    
    % requires: `CVX` (http://cvxr.com/cvx/) & `steeringreview` (https://git.io/vax96)
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025
    
    dim = d;
    oa = d; % number of outcomes
    ma = 3; % number of measurements
    Ndet = oa^ma;
    D = genSinglePartyArray(oa,ma); % generate deterministic strategy
    S_lb=[];
    for v = par
        sigma_a_x = targets.Isotropic_assemblage_3m(d,v); % change here when the target assemblage changed
        cvx_begin sdp quiet 
            variable mu_a_x(oa,ma)
            variable sigma_lambda(dim,dim,Ndet) hermitian semidefinite
            objective = 0.5/ma*sum(mu_a_x(:));
            minimize objective
            subject to
                for x=1:ma
                    for a=1:oa
                        S = zeros(dim, dim);
                        for l=1:Ndet
                            S = S + D(a,x,l)*sigma_lambda(:,:,l);
                        end
                        sigma_a_x(:,:,a,x)-S+mu_a_x(a,x)*eye(dim) == hermitian_semidefinite(dim);
                        mu_a_x(a,x)*eye(dim)+S-sigma_a_x(:,:,a,x) == hermitian_semidefinite(dim);
                    end
                end
                sum(sigma_lambda,3) == sum(sigma_a_x(:,:,:,1),3);
        cvx_end
        S_lb = [S_lb cvx_optval];
    end
end