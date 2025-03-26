classdef targets
    methods(Static)
        % `targets` generates the target assemblages
        % Input:
        %   v - visibilitiy
        % Output:
        %   sigma - the target assemblages

        % requires: nothing
        % author: Mengyan Li(mylmengyanli@gmail.com)
        % last updated: February 24, 2025
        
        function sigma = GHZ_assemblage_1unt_2m(v) % with 2 measurements
            Pauli_x = [0, 1; 1, 0];
            Pauli_z = [1, 0; 0, -1];
            sigma = zeros(4, 4, 2, 2);
            Pauli = {Pauli_x, Pauli_z};
            for x = 1:2
                for a = 1:2
                    projector = (eye(2)+ (-1)^(a-1)* Pauli{x})/2;
                    sigma(:,:,a,x) =(1-v)/8*eye(4) + v/2 * ...
                        [projector(1,1), 0, 0, projector(2,1);
                        0, 0, 0, 0;
                        0, 0, 0, 0;
                        projector(1,2), 0, 0, projector(2,2)];
                end
            end
        end
        function sigma = GHZ_assemblage_1unt_3m(v) % with 3 measurements
            Pauli_x = [0, 1; 1, 0];
            Pauli_y = [0, -1i; 1i, 0];
            Pauli_z = [1, 0; 0, -1];
            sigma = zeros(4, 4, 2, 3);   
            Pauli = {Pauli_x, Pauli_y, Pauli_z};
            for x = 1:3  
                for a = 1:2 
                    projector = (eye(2)+ (-1)^(a-1)* Pauli{x})/2;
                    sigma(:,:,a,x) =(1-v)/8*eye(4) + v/2 * ...
                        [projector(1,1), 0, 0, projector(2,1);
                        0, 0, 0, 0;
                        0, 0, 0, 0;
                        projector(1,2), 0, 0, projector(2,2)];
                end
            end
        end
    end
end