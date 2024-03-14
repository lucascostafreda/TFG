% Initial guess for pi
pi = [0.5, 0.5, 0.5];

% Optimization parameters
maxIterations = 100; % Maximum number of iterations
tolerance = 1e-4; % Tolerance for convergence
lastCost = Inf; % Initialize last cost

for iter = 1:maxIterations
    % Run FL environment with current pi
    currentCost = runFLEnvironment(pi);
    
    % Check for convergence
    if abs(lastCost - currentCost) < tolerance
        disp('Convergence achieved.');
        break;
    end
    
    % Update pi based on the observed performance
    % This is a placeholder for the actual logic to adjust pi
    % For demonstration, randomly perturb pi within the constraints
    
    
    
    % Apply constraint (simple example, adjust according to your needs)
    % This step ensures the resource constraint is met after adjusting pi
    while sum(pi .* r) > RB_available
        pi = pi * 0.99; % Scale down uniformly if constraint is violated
    end
    
    lastCost = currentCost; % Update last cost
    disp(['Iteration ', num2str(iter), ': Cost = ', num2str(currentCost)]);
end

% Final optimal pi
disp('Optimal pi:');
disp(pi);
