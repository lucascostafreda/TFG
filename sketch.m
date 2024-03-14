num_users = 10;
CQI_indices = randi([1 15], 1, num_users);
total_available_RBs = 100;
[r, ~, ~, ~] = parameters(CQI_indices);
inverse_r = 1 ./ r;
PI = inverse_r / sum(inverse_r);

% Simulation over multiple rounds
iterations = 50;
for round = 1:iterations
        available_RBs = total_available_RBs;
        selected_users = selectUsersBasedOnPI(PI, available_RBs, r);
    
        % Simulate the sending of parameters and updating of the global model here
    % ...

    % Optionally, update the PI vector based on feedback from the current round
    % ...

    fprintf('Round %d: Selected users %s\n', round, mat2str(selected_users));
end


% function selected_users = selectUsersBasedOnPI(PI, available_RBs, r)
%     % Generate random numbers for each user
%     random_values = rand(1, length(PI)); % Random number between 0 and 1
%     % simulates the randomness of network conditions or user availability at any given moment.
% 
%     % Users are selected if their random value is less than their threshold
%     is_selected = random_values < PI;
% 
%     % Calculate the actual resource usage by selected users
%     selected_resource_usage = sum(r(is_selected));
% 
%     % If selected resource usage exceeds the available RBs, deselect users randomly until within limit
%     while selected_resource_usage > available_RBs
%         % Randomly deselect one user
%         users_to_deselect = find(is_selected);
%         user_to_remove = users_to_deselect(randi(length(users_to_deselect)));
%         is_selected(user_to_remove) = false;
%         selected_resource_usage = sum(r(is_selected));
%     end
% 
%     % Return the indices of the selected users
%     selected_users = find(is_selected);
% end

% OJO, selects users based on their resource efficiency or another metric 
% until adding another user would exceed the limit.

function selected_users = selectUsersGreedy(users, available_RBs, r)
    % Sort users by their resource requirements (ascending order for this example)
    [sorted_r, sorted_indices] = sort(r, 'ascend');

    selected_users = [];
    current_RBs = 0;

    for i = 1:length(users)
        user_idx = sorted_indices(i);
        if (current_RBs + sorted_r(i)) <= available_RBs
            % Add user if total RBs won't exceed the limit
            selected_users = [selected_users, user_idx];
            current_RBs = current_RBs + sorted_r(i);
        else
            % If adding the next user exceeds the limit, break the loop
            break;
        end
    end
end

function selected_users = optimizeSelection(selected_users, r, available_RBs)
    current_usage = sum(r(selected_users));
    remaining_RBs = available_RBs - current_usage;
    all_users = 1:length(r);
    unselected_users = setdiff(all_users, selected_users);
    
    % Sort unselected users by how much they 'fill up' the remaining RBs, descending
    [sorted_diffs, sort_idx] = sort(r(unselected_users) - remaining_RBs, 'descend');
    sorted_unselected_users = unselected_users(sort_idx);
    
    for i = 1:length(sorted_unselected_users)
        user_to_consider = sorted_unselected_users(i);
        if r(user_to_consider) <= remaining_RBs
            % If adding this user does not exceed the total RBs, add them
            selected_users = [selected_users, user_to_consider];
            remaining_RBs = remaining_RBs - r(user_to_consider);
            if remaining_RBs == 0
                % Perfectly utilized all RBs
                break;
            end
        end
    end
    
    % Attempt to swap if possible (very basic swap logic for illustration)
    if remaining_RBs > 0
        for i = 1:length(selected_users)
            for j = 1:length(sorted_unselected_users)
                if r(sorted_unselected_users(j)) - r(selected_users(i)) <= remaining_RBs
                    % Perform a swap if it increases utilization without exceeding limit
                    temp_selection = selected_users;
                    temp_selection(i) = sorted_unselected_users(j); % Swap
                    new_usage = sum(r(temp_selection));
                    if new_usage <= available_RBs && new_usage > current_usage
                        selected_users = temp_selection; % Accept swap
                        current_usage = new_usage;
                        remaining_RBs = available_RBs - new_usage;
                        break; % Break inner loop
                    end
                end
            end
        end
    end
end


% dynamically redifining CQI for each user


% num_users = 10; % Number of users
% total_available_RBs = 100; % Total available RBs in the network
% num_rounds = 50; % Number of iterations to simulate
% 
% for round = 1:num_rounds
%     % Dynamically redefine CQI for each user
%     CQI_indices = randi([1, 15], 1, num_users); % Random CQI indices for each user
% 
%     % Call the 'parameters' function to get resource allocation for each user
%     [r, ~, ~, ~] = parameters(CQI_indices);
% 
%     % Recalculate the probability vector 'PI' based on new 'r'
%     inverse_r = 1 ./ r; % Inverse the resource allocation
%     PI = inverse_r / sum(inverse_r); % Normalize to create the probability vector 'PI'
% 
%     % Select users based on updated PI and resource constraints
%     selected_users = selectUsersBasedOnPI(PI, total_available_RBs, r);
% 
%     % Simulate the sending of parameters and updating of the global model here
%     % ...
% 
%     fprintf('Round %d: Selected users %s\n', round, mat2str(selected_users));
% end

function selected_users = selectUsersBasedOnPI(PI, available_RBs, r, num_users)
    % Initialize an empty array for selected users
    selected_users = [];
    
    % Calculate the total resource usage
    total_resource_usage = 0;
    
    % Keep selecting users until the resource limit is reached
    while total_resource_usage < available_RBs
        % Select one user based on PI probabilities
        user = randsrc(1, 1, [1:num_users; PI]);
        
        % Check if this user is already selected
        if ismember(user, selected_users)
            continue; % If already selected, skip to the next iteration
        end
        
        % Check if adding this user exceeds the available RBs
        if total_resource_usage + r(user) <= available_RBs
            % Add the user to the selected list
            selected_users = [selected_users, user];
            
            % Update the total resource usage
            total_resource_usage = total_resource_usage + r(user);
        else
            % If adding any more users exceeds the limit, break the loop
            break;
        end
    end
end

% RESOURCE MAXIMIZATION
% es erroneo, es, o minimizamos el consumo sabiendo el error de estimación
% o minimizamos el error de estimación dado un límite de recursos

% The optimized PI is in PI_optimized and maximizes the sum(PI * r) 
% without exceeding the total_available_RBs
f = -r; % Coefficients for the objective function (negative for maximization)

% Inequality constraints: sum(PI * r) <= total_available_RBs
% This is represented as A * PI <= b in linprog
A = r; % Coefficients matrix for the inequality constraints
b = total_available_RBs; % Right-hand side vector for the inequality constraints

% Bounds for the selection probabilities PI
lb = zeros(usernumber, 1); % Lower bound (PI >= 0)
ub = ones(usernumber, 1); % Upper bound (PI <= 1)

% Options: Using 'dual-simplex' algorithm for the optimization
options = optimoptions('linprog', 'Algorithm', 'dual-simplex');

% Solve the linear programming problem
[PI_optimized, ~, exitflag, ~] = linprog(f, A, [], [], [], [], lb, ub, options);

% Check if the solution was found successfully
if exitflag == 1
    fprintf('Optimized PI found successfully.\n');
    fprintf('Optimized PI values: %s\n', mat2str(PI_optimized, 4));
else
    fprintf('Solution not found. Exit flag: %d\n', exitflag);
end

% MINIMIZATION SERVICE COST  FUNCTIO
% Load gradient data for constrained and reference models
load('constrained_gradient_vector.mat', 'totalGradientConstrained');
load('reference_gradient_vector.mat', 'totalGradientReference');

% Initialize policy PI and other parameters
PI = initial_PI; % Assuming initial_PI is defined based on your system's constraints
learningRate = 0.01; % Learning rate for PI updates
iterations = 100; % Define the number of iterations for the optimization process

for iter = 1:iterations
    % Calculate the cost service function for the current PI
    % Assuming a simple absolute difference cost function
    costServiceFunction = abs(totalGradientConstrained - totalGradientReference);
    
    % Calculate the gradient of the cost service function w.r.t. PI
    % This is a placeholder; the actual calculation will depend on your system
    gradPI = calculateGradientPI(PI, costServiceFunction);
    
    % Update PI using gradient descent
    PI = PI - learningRate * gradPI;
    
    % Ensure PI meets your system's constraints (e.g., sum(PI*r) <= R)
    PI = enforceConstraints(PI);
    
    % Optionally, update learningRate or apply other optimizations
    
    % Evaluate and print the cost service function's progress
    fprintf('Iteration %d, Cost: %f\n', iter, sum(costServiceFunction));
end

% Save the optimized PI for further use
save('optimized_PI.mat', 'PI');


% Initialize parameters
PI = inverse_r / sum(inverse_r); % Initial PI based on your setup
learning_rate = 0.01; % Learning rate for PI updates
maxIterations = 100; % Maximum number of iterations for the optimization
tolerance = 1e-4; % Tolerance for stopping criteria based on service cost change
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
for iteration = 1:maxIterations
    % Train local models and compute deviations from reference model
    % This part would involve your existing FL training and evaluation code
    
    % Placeholder for service cost computation - this needs actual implementation
    serviceCost = computeServiceCost(deviationW, deviationb, PI, r, RBs);
    
    % Compute or approximate the gradient of the service cost w.r.t. PI
    % This is a critical step that requires a method to compute or approximate gradients
    gradientPI = approximateGradientPI(PI, r, RBs);
    
    % Update PI using the gradient
    PI = PI - learning_rate * gradientPI;
    
    % Ensure PI remains a valid probability distribution
    PI = max(PI, 0); % Ensure non-negative
    PI = PI / sum(PI); % Normalize to sum to 1
    
    % Check if the sum(PI * r) exceeds available RBs, adjust if necessary
    if sum(PI .* r) > RBs
        % Adjust PI to meet the constraint - this is a simplification
        % More sophisticated projection methods might be required
        PI = adjustPI(PI, r, RBs);
    end
    
    % Compute new service cost after update
    newServiceCost = computeServiceCost(deviationW, deviationb, PI, r, RBs);
    
    % Check for convergence
    if abs(newServiceCost - serviceCost) < tolerance
        disp('Convergence achieved.');
        break;
    end
    
    % Optionally, adjust learning_rate or perform other updates
end

function gradPI = approximateGradientPI(PI, r, RBs)
    % Placeholder for gradient approximation
    % Implement a method to approximate the gradient of the service cost w.r.t. PI
    gradPI = zeros(size(PI)); % This needs to be replaced with actual computation
end

function PI = adjustPI(PI, r, RBs)
    % Placeholder for adjusting PI to meet the RBs constraint
    % Implement a method to adjust PI so that sum(PI * r) <= RBs
    PI = PI * (RBs / sum(PI .* r)); % This is a simplification
end

function cost = computeServiceCost(deviationW, deviationb, PI, r, RBs)
    % Placeholder for service cost computation
    % Implement the actual service cost computation based on deviations and PI
    cost = sum(abs(deviationW)) + sum(abs(deviationb)); % Simplification
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example parameters
NUM_DEVICES = 10; % Number of devices/users
TOTAL_RBS = 100; % Total RBs available
r = rand(1, NUM_DEVICES); % Example RB distribution among devices
W_ref = rand(1, NUM_DEVICES); % Reference model weights for demonstration
W_pi = rand(1, NUM_DEVICES); % Placeholder for the actual Wpi calculation

% GA options
options = optimoptions('ga', 'Display', 'iter', 'UseParallel', false, 'PopulationSize', 50, 'MaxGenerations', 100);

% Objective function wrapper to include additional parameters
objFunc = @(PI) objectiveFunction(PI, W_pi, W_ref);

% Constraint function wrapper
nonlcon = @(PI) constraintFunction(PI, r, TOTAL_RBS);

% Running the GA: Assuming PI values are between 0 and 1
[PI_opt, cost_opt] = ga(objFunc, NUM_DEVICES, [], [], [], [], zeros(1, NUM_DEVICES), ones(1, NUM_DEVICES), nonlcon, options);

fprintf('Optimal PI: %s\n', mat2str(PI_opt));
fprintf('Minimum Cost: %f\n', cost_opt);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Before the iteration loop starts, load or define the parameters of the 
% reference global model. This model serves as the benchmark for optimizing PI.
load('referenceGlobalModel.mat', 'refGlobalW1', 'refGlobalW2', 'refGlobalW3', 'refGlobalW4', 'refGlobalW5', 'refGlobalB1', 'refGlobalB2', 'refGlobalB3', 'refGlobalB4', 'refGlobalB5');

% Inside the for-loop, after updating the constrained global model


% Assuming globalParams and refGlobalParams are already defined
usernumber = numel(PI); % Number of elements in PI
r = ... % RB distribution among devices
totalRBs = ... % Total available RBs

% Initial guess for PI
PI_initial = ones(1, usernumber) * (totalRBs / sum(r)); % Equal distribution considering resources

% Define linear inequality constraints (A*x <= b) for PI bounds
A = []; b = []; % No linear inequality constraints
Aeq = []; beq = []; % No linear equality constraints
lb = zeros(1, usernumber); % Lower bound of PI (0 for each element)
ub = ones(1, usernumber); % Upper bound of PI (1 for each element)

% Nonlinear constraints
nonlcon = @(PI)constraints(PI, r, totalRBs);

% Optimization options
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

% Objective function
objective = @(PI)objectiveFunction(PI, globalParams, refGlobalParams, r, totalRBs);

% Solve the optimization problem
[PI_opt, deviation_opt] = fmincon(objective, PI_initial, A, b, Aeq, beq, lb, ub, nonlcon, options);


function [c, ceq] = constraints(PI, r, totalRBs)
    c = sum(PI .* r) - totalRBs; % Resource constraint: Should be <= 0
    ceq = []; % No equality constraints
end

function deviation = objectiveFunction(PI, globalParams, refGlobalParams, r, totalRBs)
    % Calculate deviation
    deviation = norm(globalParams - refGlobalParams, 2);
    % Incorporate additional logic if necessary to use PI in the calculation
end

function FLTrainingWithAveragedParams(categories, rootFolderTrain, rootFolderTest, iterations, usernumber)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imds_test = imageDatastore(fullfile(rootFolderTest, categories), 'LabelSource', 'foldernames');
    imds_train = imageDatastore(fullfile(rootFolderTrain, categories), 'LabelSource', 'foldernames');
    
    % Split dataset for IID distribution among users
    [imds1, imds2, imds3, imds4, imds5, imds6, imds7, imds8] = splitEachLabel(imds_train, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125);

    % FL Training setup
    varSize = 32; % Image dimension
    learningRate = 0.008;

    % Initialize a figure for plotting accuracy
    figure;
    hold on;
    xlabel('Iteration');
    ylabel('Accuracy (%)');
    title('Real-Time Accuracy Plot');
    grid on;
    
    accuracies = zeros(1, iterations);

    % Load averaged parameters for all iterations
    filename = sprintf('Averaged_Ref_Model_Params_%d_Iterations.mat', iterations);
    if isfile(filename)
        load(filename, 'avgParams');
    else
        error('Averaged parameters file %s not found.', filename);
    end

    for i = 1:iterations
        fprintf('Starting FL iteration %d\n', i);
       % Define the CNN architecture with averaged parameters
        layers = [
            imageInputLayer([varSize varSize 3], 'Name', 'input')
            convolution2dLayer(5, 32, 'Padding', 2, 'Name', 'conv1', 'Weights', avgParams(i).globalw1, 'Bias', avgParams(i).globalb1)
            reluLayer('Name', 'relu1')
            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
            convolution2dLayer(5, 32, 'Padding', 2, 'Name', 'conv2', 'Weights', avgParams(i).globalw2, 'Bias', avgParams(i).globalb2)
            reluLayer('Name', 'relu2')
            averagePooling2dLayer(2, 'Stride', 2, 'Name', 'avgpool1')
            convolution2dLayer(5, 64, 'Padding', 2, 'Name', 'conv3', 'Weights', avgParams(i).globalw3, 'Bias', avgParams(i).globalb3)
            reluLayer('Name', 'relu3')
            fullyConnectedLayer(64, 'Name', 'fc1', 'Weights', avgParams(i).globalw4, 'Bias', avgParams(i).globalb4)
            reluLayer('Name', 'relu4')
            fullyConnectedLayer(numel(categories), 'Name', 'fc2', 'Weights', avgParams(i).globalw5, 'Bias', avgParams(i).globalb5)
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];

        % Training options remain the same
        options = trainingOptions('adam', ...
            'InitialLearnRate', learningRate, ...
            'MaxEpochs', 1, ...
            'MiniBatchSize', 50, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false);

        % Example training loop for each user (simplified)
        for user = 1:usernumber
            eval(sprintf('userDataset = imds%d;', user)); % Dynamically select user's dataset
            [trainedNet, trainInfo] = trainNetwork(userDataset, layers, options);

            % Here, one might aggregate updates from users, but we're applying averaged parameters directly
        end

        % Calculate identification accuracy for the iteration using the test dataset
        predictedLabels = classify(trainedNet, imds_test);
        accuracy = sum(predictedLabels == imds_test.Labels) / numel(imds_test.Labels);
        accuracies(i) = accuracy * 100; % Store accuracy in percentage
        
        fprintf('Accuracy for iteration %d: %.2f%%\n', i, accuracies(i));
        
        % Update the plot with the new accuracy value
        plot(1:i, accuracies(1:i), '-o', 'LineWidth', 2);
        drawnow; % Ensure the plot updates in real-time
    end
    hold off; %release the figure
end

% HOW TO CALL THE FUNCTION FROM THECOMMAND WINDOW
categories = {'deer', 'dog', 'frog', 'cat', 'bird', 'automobile', 'horse', 'ship', 'truck', 'airplane'};
rootFolderTrain = 'cifar10Test';
rootFolderTest = 'cifar10Train';
iterations = 10; % For example, 10 iterations
usernumber = 8; % For example, 8 users

FLTrainingWithAveragedParams(categories, rootFolderTrain, rootFolderTest, iterations, usernumber);

layer = [
    imageInputLayer([varSize varSize 3]); %input layer with the size of the input images
    convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2); %2D convolutional layer with 5x5 filters.
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    fullyConnectedLayer(64,'BiasLearnRateFactor',2); 
    reluLayer();
    fullyConnectedLayer(length(categories),'BiasLearnRateFactor',2);
    softmaxLayer()
    classificationLayer()];

option = trainingOptions('adam', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 8, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 50, ...
    'ExecutionEnvironment','parallel',...
    'Shuffle', 'every-epoch',... % mezclar antes de cada epoch, no sirve de mucho, pues pasasuna vez
    'Verbose', true);%,... 
    %'Plots', 'training-progress'); 
%%%%%%%%%%%%%%%%%%%%%%%%% FIGURES INIT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize the figure for plotting
fig1=figure;
hold on; 
 
xlabel('Iteration');
ylabel('Current Accuracy');
title('Real-Time Accuracy Plot (RM_NE)');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FL START %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:1:iteration

for user=1:1:usernumber  
    
    clear netvaluable;
    Winstr1=strcat('net',int2str(user));     
    midstr=strcat('imds',int2str(user)); 
    %creates a string that refers to the ImageDatastore specific to the current user.
     
    eval(['imdss','=',midstr,';']);
    % string as MATLAB code, assigning the user-specific ImageDatastore to the variable imdss
    % allows selecting the correct dataset for each user during the iteration.

 if i>1 

    layer(2).Weights=globalw1;
    layer(5).Weights=globalw2;
    layer(8).Weights=globalw3;
    layer(11).Weights=globalw4;
    layer(13).Weights=globalw5;  

    layer(2).Bias=globalb1;    
    layer(5).Bias=globalb2;
    layer(8).Bias=globalb3;
    layer(11).Bias=globalb4;
    layer(13).Bias=globalb5;

end

% Proceed with training if the subset is valid
[netvaluable, info] = trainNetwork(imdss, layer, option);
