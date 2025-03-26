classdef targets
    methods(Static)
        % `targets` generates the target assemblages
        % Input:
        %   d - dimension of the system
        %   v - visibilitiy
        % Output:
        %   sigma - the target assemblages

        % requires: nothing
        % author: Mengyan Li(mylmengyanli@gmail.com)
        % last updated: February 24, 2025

        function sigma = Isotropic_assemblage(d,v) % with d+1 measurements
            if d==4
                load('mubs_4.mat');
            else
                mubs = MUBs(d);
            end
            sigma = zeros(d,d,d,d+1);
            for x=1:d+1
                for a = 1:d
                    sigma(:,:,a,x)= v/d*transpose(mubs{x,a}*mubs{x,a}') +(1-v)/(d^2)*eye(d);
                end
            end
        end
                
        function sigma = Isotropic_assemblage_2m(d,v) % with 2 measurements
            if d==4
                load('mubs_4.mat');
                mubs = mubs(1:2,:);
            else
                mubs = MUBs(d);
                mubs = mubs(1:2,:);
            end
            sigma = zeros(d,d,d,2);
            for x=1:2
                for a = 1:d
                    sigma(:,:,a,x)= v/d*transpose(mubs{x,a}*mubs{x,a}') +(1-v)/(d^2)*eye(d);
                end
            end
        end

        function sigma = Isotropic_assemblage_3m(d,v) % with 3 measurements
            if d==4
                load('mubs_4.mat');
                mubs = mubs(1:3,:);
            else
                mubs = MUBs(d);
                mubs = mubs(1:3,:);
            end
            sigma = zeros(d,d,d,3);
            for x=1:3
                for a = 1:d
                    sigma(:,:,a,x)= v/d*transpose(mubs{x,a}*mubs{x,a}') +(1-v)/(d^2)*eye(d);
                end
            end
        end
    end
end