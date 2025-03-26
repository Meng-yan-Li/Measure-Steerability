function S_ub = upper_bound(par,oa,ma)
    % `upper_bound` determines the upper bound of multipartite quantum steerability with 1-UNT
    % Input:
    %   par - visibility parameter of the target assemblage
    %   oa - number of outcomes
    %   ma - number of measurements
    % Output:
    %   S_ub - upper bound of multipartite quantum steerability with 1-UNT
    
    % requires: `CVX` (http://cvxr.com/cvx/)
    %           `steeringreview` (https://git.io/vax96)
    %           `QETLAB` (http://www.qetlab.com)
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025
    
    dim = 2;
    k=2;
    dB = dim;
    dC = dim;
    Ndet = oa^ma;
    D = genSinglePartyArray(oa,ma); % generate deterministic strategy
    S_ub=[];
    for v = par
        sigma_a_x = targets.GHZ_assemblage_1unt_2m(v);% change here when the target assemblage changed
        cvx_begin sdp quiet
            variable sigma_lambda(dim^2,dim^2,Ndet) hermitian semidefinite
            variable pi_a_x(dim^2,dim^2,oa,ma) hermitian semidefinite
            objective = trace(sum(sigma_lambda,3))-1;
            minimize objective
            subject to
                for x=1:ma
                    for a = 1:oa
                        S=zeros(dim^2, dim^2);
                        for l=1:Ndet
                            S=S+D(a,x,l).*sigma_lambda(:,:,l);
                        end
                        S-sigma_a_x(:,:,a,x)==pi_a_x(:,:,a,x);
                    end
                    sum(pi_a_x(:,:,:,x),3) == (trace(sum(sigma_lambda,3))-1).*(sigma_a_x(:,:,1,1)+sigma_a_x(:,:,2,1));
                end
                for l=1:Ndet
                    % k-symmetric PPT extendible states as an outer approximation to the set of separable states.
                    SymmetricExtension(sigma_lambda(:,:,l),k,[dB,dC],1,1) == 1; % separable constraint 
                end
                SymmetricExtension(sum(sigma_lambda,3),k,[dB,dC],1,1) == 1; 
                trace(sum(sigma_lambda,3))-1>=0;
        cvx_end
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
