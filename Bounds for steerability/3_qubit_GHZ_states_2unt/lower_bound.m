function S_lb = lower_bound(par,oa,ob,ma,mb)
    % `lower_bound` determines the lower bound of multipartite quantum steerability with 2-UNT
    % Input:
    %   par - visibility parameter of the target assemblage
    %   oa - number of outcomes of A_1
    %   ob - number of outcomes of A_2
    %   ma - number of measurements of A_1
    %   mb - number of measurements of A_2
    % Output:
    %   S_lb - lower bound of multipartite quantum steerability with 2-UNT
    
    % requires: `CVX` (http://cvxr.com/cvx/) & `steeringreview` (https://git.io/vax96)
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025
    
    dim = 2;
    Ndet_A = oa^ma;
    Ndet_B = ob^mb;
    D_A = genSinglePartyArray(oa,ma); % generate deterministic strategy
    D_B = genSinglePartyArray(ob,mb); 
    S_lb=[];
    for v = par
        sigma_a_b_x_y = targets.GHZ_assemblage_2unt_3m(v); % change here when the target assemblage changed
        cvx_begin sdp quiet
            variable mu_a_b_x_y(oa,ob,ma,mb)
            variable sigma_lambda(dim,dim,Ndet_A,Ndet_B) hermitian semidefinite
            objective = 0.5/(ma*mb)*sum(mu_a_b_x_y(:));
            minimize objective
            subject to
                for y=1:mb
                    for x=1:ma
                        for b=1:ob
                            for a = 1:oa
                                S = zeros(dim, dim);
                                for l=1:Ndet_A
                                    for v=1:Ndet_B
                                         S = S + D_A(a,x,l)*D_B(b,y,v)*sigma_lambda(:,:,l,v);
                                    end
                                end
                                sigma_a_b_x_y(:,:,a,b,x,y)-S+mu_a_b_x_y(a,b,x,y)*eye(dim) == hermitian_semidefinite(dim);
                                mu_a_b_x_y(a,b,x,y)*eye(dim)+S-sigma_a_b_x_y(:,:,a,b,x,y) == hermitian_semidefinite(dim);
                            end
                        end
                    end
                end
                sum(sum(sigma_lambda, 3), 4)==sigma_a_b_x_y(:,:,1,1,1,1)+sigma_a_b_x_y(:,:,1,2,1,1)+sigma_a_b_x_y(:,:,2,1,1,1)+sigma_a_b_x_y(:,:,2,2,1,1);
        cvx_end
        S_lb = [S_lb cvx_optval];
    end
end