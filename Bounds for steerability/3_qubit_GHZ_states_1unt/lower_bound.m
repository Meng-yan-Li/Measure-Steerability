function S_lb = lower_bound(par,oa,ma)
    % `lower_bound` determines the lower bound of multipartite quantum steerability with 1-UNT
    % Input:
    %   par -visibility parameter of the target assemblage
    %   oa - number of outcomes
    %   ma - number of measurements
    % Output:
    %   S_lb - lower bound of multipartite quantum steerability with 1-UNT
    
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
    S_lb=[];
    for v = par
        sigma_a_x = targets.GHZ_assemblage_1unt_2m(v); % change here when the target assemblage changed
        cvx_begin sdp quiet
            variable mu_a_x(oa,ma)
            variable sigma_lambda(dim^2,dim^2,Ndet) hermitian semidefinite
            objective = 0.5/ma*sum(mu_a_x(:));
            minimize objective
            subject to
                for x=1:ma
                    for a=1:oa
                        S = zeros(dim^2, dim^2);
                        for l=1:Ndet
                            S = S + D(a,x,l)*sigma_lambda(:,:,l);
                        end
                        sigma_a_x(:,:,a,x)-S+mu_a_x(a,x)*eye(dim^2) == hermitian_semidefinite(dim^2);
                        mu_a_x(a,x)*eye(dim^2)+S-sigma_a_x(:,:,a,x) == hermitian_semidefinite(dim^2);
                    end
                end
                for l=1:Ndet
                    % k-symmetric PPT extendible states as an outer approximation to the set ofseparable states.
                    SymmetricExtension(sigma_lambda(:,:,l),k,[dB,dC],1,1) == 1; % separable constraint 
                end
                SymmetricExtension(sum(sigma_lambda,3),k,[dB,dC],1,1) == 1; 
                sum(sigma_lambda,3) == sigma_a_x(:,:,1,1)+sigma_a_x(:,:,2,1);
                
        cvx_end
        S_lb = [S_lb cvx_optval];
    end
end