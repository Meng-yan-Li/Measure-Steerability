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
        
        function sigma = GHZ_assemblage_2unt_2m(v)
            Pauli_x = [0, 1; 1, 0];
            Pauli_z = [1, 0; 0, -1];
            sigma = zeros(2, 2, 2, 2, 2, 2);
            diagonal = diag([(1-v)/8, (1-v)/8]);
            Pauli = {Pauli_x, Pauli_z};
            for x = 1:2
                for a = 1:2
                    for y = 1:2
                        for b = 1:2
                            projector_A=(eye(2)+(-1)^a*Pauli{x})/2;
                            projector_B=(eye(2)+(-1)^b*Pauli{y})/2;
                            sigma(:,:,a,b,x,y) = diagonal + v/2 *...
                            [projector_A(1,1)*projector_B(1,1) projector_A(2,1)*projector_B(2,1);
                             projector_A(1,2)*projector_B(1,2) projector_A(2,2)*projector_B(2,2)];
                        end
                    end
                end
            end
        end
        function sigma = GHZ_assemblage_2unt_3m(v)
            Pauli_x = [0, 1; 1, 0];
            Pauli_y = [0, -1i; 1i, 0];
            Pauli_z = [1, 0; 0, -1];
            sigma = zeros(2, 2, 2, 2, 3, 3);
            diagonal = diag([(1-v)/8, (1-v)/8]);
            Pauli = {Pauli_x, Pauli_y, Pauli_z};
            for x = 1:3
                for a = 1:2
                    for y = 1:3
                        for b = 1:2
                            projector_A=(eye(2)+(-1)^a*Pauli{x})/2;
                            projector_B=(eye(2)+(-1)^b*Pauli{y})/2;
                            sigma(:,:,a,b,x,y) = diagonal + v/2 *...
                            [projector_A(1,1)*projector_B(1,1) projector_A(2,1)*projector_B(2,1);
                             projector_A(1,2)*projector_B(1,2) projector_A(2,2)*projector_B(2,2)];
                        end
                    end
                end
            end
        end
    end
end