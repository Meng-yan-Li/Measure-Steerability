function mubs = MUBs(d)
    % `MUBs` returns the mubs in d dimension
    % Input:
    %   d - dimension of the system
    % Output:
    %   mubs - mutually unbiased bases in d dimension
    
    % requires: nothing
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025
    
    FourierBase = {};
    CompuBase = {};
    if isprime(d)==0 || d==2
        if fix(log2(d))==log2(d)
            for x=0:d-1
                OneHot = zeros(d,1);
                OneHot(x+1)=1;
                Base_x = {};
                for a=0:d-1
                    Base_a = [];
                    for l=0:d-1
%                         Base_a = [Base_a;1i^((x+2*a)*l)];
                        Base_a = [Base_a;exp(1i*pi/2*absolute_trace(x+2*a,2,log2(d))*l)];
                    end
                    Base_a = Base_a/sqrt(d);
                    Base_x = [Base_x Base_a];
                end
                CompuBase = [CompuBase OneHot];
                FourierBase = [FourierBase;Base_x] ;
            end
            mubs = [CompuBase;FourierBase];
        else
            return
        end
    else
        for x=0:d-1
            OneHot = zeros(d,1);
            OneHot(x+1)=1;
            Base_x = {};
            for a=0:d-1
                Base_a = [];
                for l=0:d-1
                    Base_a = [Base_a;exp(1i*2*pi/d*(a*l+x*l^2))];
                end
                Base_a = Base_a/sqrt(d);
                Base_x = [Base_x Base_a];
            end
            CompuBase = [CompuBase OneHot];
            FourierBase = [FourierBase;Base_x] ;
        end
        mubs = [CompuBase;FourierBase];
    end
end

