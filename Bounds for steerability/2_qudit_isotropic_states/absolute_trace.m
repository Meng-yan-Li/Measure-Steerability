function trace = absolute_trace(x, p, m)
    % `absolute_trace` computes the trace from Fq to Fp
    % Input:
    %   x - an element of Fq (given as an integer representation)
    %   p - the characteristic of the prime field Fp
    %   m - the degree of the field extension Fq = Fp^m
    % Output:
    %   trace - the absolute trace of x in Fp
    
    % requires: nothing
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025

    % Initialize the trace
    trace = 0;
    % Compute the trace: tr(x) = x + x^p + x^(p^2) + ... + x^(p^(m-1))
    for i = 0:m-1
        trace = trace+x^(p^i);
%         trace = mod(trace + mod(x^(p^i), p^m), p); % Ensure result stays in Fp
    end
end
