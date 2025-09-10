function maxEig = maxEig(EHE, max_iterations)

    % Set default value for max_iterations if not provided
    if nargin < 2
        max_iterations = 1000;
    end

    n = size(EHE, 1);
    v = rand(n, 1); % Initial random vector
    v = v / norm(v); % Normalize the vector

    tolerance = 1e-6;
    lambda_old = 0;

    for k = 1:max_iterations
        w = EHE * v; % Matrix-vector multiplication
        v = w / norm(w); % Normalize the vector
        lambda = real(v' * EHE * v); % Rayleigh quotient

        if abs(lambda - lambda_old) < tolerance
            break;
        end
        lambda_old = lambda;
    end

    maxEig = lambda;
end