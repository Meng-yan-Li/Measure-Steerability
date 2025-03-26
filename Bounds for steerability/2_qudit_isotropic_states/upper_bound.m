function S_ub = upper_bound(par,d)
    % `upper_bound` determines the upper bound of bipartite quantum steerability
    % Input:
    %   par - visibility parameter of the target assemblage
    %   d - dimension of the system
    % Output:
    %   S_ub - upper bound of bipartite quantum steerability
    
    % requires: `CVX` (http://cvxr.com/cvx/) & `steeringreview` (https://git.io/vax96)
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025
    
    dim = d;
    oa = d; % number of outcomes
    ma = 3; % number of measurements
    Ndet = oa^ma;
    D = genSinglePartyArray(oa,ma); % generate deterministic strategy
    S_ub = [];
    for v = par
        sigma_a_x = targets.Isotropic_assemblage_3m(d,v); % change here when the target assemblage changed
        cvx_begin sdp quiet 
            variable sigma_lambda(dim,dim,Ndet) hermitian semidefinite
            variable pi_a_x(dim,dim,oa,ma) hermitian semidefinite
            objective = trace(sum(sigma_lambda,3))-1;
            minimize objective
            subject to
                for x=1:ma
                    for a = 1:oa
                        S=zeros(dim, dim);
                        for l=1:Ndet
                            S=S+D(a,x,l).*sigma_lambda(:,:,l);
                        end
                        S-sigma_a_x(:,:,a,x)==pi_a_x(:,:,a,x);
                    end
                    sum(pi_a_x(:,:,:,x),3) == (trace(sum(sigma_lambda,3))-1).*sum(sigma_a_x(:,:,:,1),3);
                end
                trace(sum(sigma_lambda,3))-1>=0;
        cvx_end
    %     rho_a_x{end + 1} = (pi_a_x + sigma_a_x) / (1 + cvx_optval);
        rho_a_x = (pi_a_x + sigma_a_x) ./ (1 + cvx_optval); % construct a new assemblage
        Dis=0;
        for x=1:ma
            for a = 1:oa
                Dis=Dis+1/ma*trace_distance(sigma_a_x(:,:,a,x),rho_a_x(:,:,a,x));
            end
        end
        S_ub = [S_ub Dis];
    end
end
