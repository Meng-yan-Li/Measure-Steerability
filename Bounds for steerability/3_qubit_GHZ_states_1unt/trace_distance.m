function d = trace_distance(p, q)
    % `trace_distance` calculates the trace distance between two matrices p and q
    
    % requires: nothing
    % author: Mengyan Li(mylmengyanli@gmail.com)
    % last updated: February 24, 2025

    % Calculate the eigenvalues of the matrix (p - q)
    e = eig(p - q);
    % Compute the trace distance
    d = 0.5 * sum(abs(e));
end