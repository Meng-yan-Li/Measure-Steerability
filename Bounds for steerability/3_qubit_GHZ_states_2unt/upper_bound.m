function S_ub = upper_bound(par,oa,ob,ma,mb)
    % `upper_bound` determines the upper bound of multipartite quantum steerability with 2-UNT
    % Input:
    %   par - visibility parameter of the target assemblage
    %   oa - number of outcomes of A_1
    %   ob - number of outcomes of A_2
    %   ma - number of measurements of A_1
    %   mb - number of measurements of A_2
    % Output:
    %   S_ub - upper bound of multipartite quantum steerability with 2-UNT
    
    % requires: `CVX` (http://cvxr.com/cvx/) & `steeringreview` (https://git.io/vax96)
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025

    dim = 2;
    Ndet_A = oa^ma;
    Ndet_B = ob^mb;
    D_A = genSinglePartyArray(oa,ma); % generate deterministic strategy
    D_B = genSinglePartyArray(ob,mb);
    S_ub=[];
    for v = par
        sigma_a_b_x_y = targets.GHZ_assemblage_2unt_3m(v);% Change here when the target assemblage changed.
        cvx_begin sdp quiet
        variable sigma_lambda(dim,dim,Ndet_A,Ndet_B) hermitian semidefinite
        variable pi_a_b_x_y(dim,dim,oa,ob,ma,mb) hermitian semidefinite
        objective = trace(sum(sum(sigma_lambda,3),4))-1;
        minimize objective
        subject to
        for y=1:mb
            for x=1:ma
                for b=1:ob
                    for a = 1:oa
                        S=zeros(dim, dim);
                        for l=1:Ndet_A
                            for v=1:Ndet_B
                                S = S + D_A(a,x,l)*D_B(b,y,v)*sigma_lambda(:,:,l,v);
                            end
                        end
                        S-sigma_a_b_x_y(:,:,a,b,x,y)==pi_a_b_x_y(:,:,a,b,x,y);
                    end
                end
                sum(sum(pi_a_b_x_y(:,:,:,:,x,y),3),4) == (trace(sum(sum(sigma_lambda,3),4))-1).*(sigma_a_b_x_y(:,:,1,1,1,1)+sigma_a_b_x_y(:,:,1,2,1,1)+sigma_a_b_x_y(:,:,2,1,1,1)+sigma_a_b_x_y(:,:,2,2,1,1));
            end
        end
        trace(sum(sum(sigma_lambda,3),4))-1 >=0;
        cvx_end
        rho_a_b_x_y = (pi_a_b_x_y + sigma_a_b_x_y) ./ (1 + cvx_optval); % construct a new assemblage
        Dis=0;
        for y=1:mb
            for x=1:ma
                for b=1:ob
                    for a = 1:oa
                        Dis=Dis+1/(ma*mb)*trace_distance(sigma_a_b_x_y(:,:,a,b,x,y),rho_a_b_x_y(:,:,a,b,x,y));
                    end
                end
            end
        end
        S_ub = [S_ub Dis];
    end
end
