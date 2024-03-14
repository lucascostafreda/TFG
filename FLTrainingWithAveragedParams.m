function FLTrainingWithAveragedParams(categories, rootFolderTrain, rootFolderTest, iterations, usernumber,repetitions)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imds_test = imageDatastore(fullfile(rootFolderTest, categories), 'LabelSource', 'foldernames');
    imds_train = imageDatastore(fullfile(rootFolderTrain, categories), 'LabelSource', 'foldernames');
    
    % Split dataset for IID distribution among users
    [imds1, imds2, imds3, imds4, imds5, imds6, imds7, imds8] = splitEachLabel(imds_train, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125);

    % FL Training setup
    varSize = 32; % Image dimension

    % Initialize a figure for plotting accuracy
    figure;
    hold on;
    xlabel('Iteration');
    ylabel('Accuracy (%)');
    title('Real-Time Accuracy Plot');
    grid on;
    
    accuracies = zeros(1, iterations);

    % Load averaged parameters for all iterations
    filename = sprintf('Ref_Model_%d_i_%d_r.mat', iterations,repetitions);
    if isfile(filename)
        load(filename, 'avgParams');
    else
        error('Averaged parameters file %s not found.', filename);
    end

    % Training options remain unchanged...
    options = trainingOptions('adam', ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.2, ...
        'LearnRateDropPeriod', 8, ...
        'MaxEpochs', 1, ...
        'MiniBatchSize', 50, ...
        'ExecutionEnvironment', 'parallel', ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true);


    for i = 1:iterations
        fprintf('Starting FL iteration %d\n', i);

    % Define the initial network architecture or update it with the averaged parameters
        if i == 1
            % Define the network architecture for the first iteration
            layers = [
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
    
        else
            % Update the layers with averaged parameters from avgParams for subsequent iterations
            % Indices of layers with weights and biases
            layers(2).Weights = avgParams(i).(['globalw', num2str(1)]);
            layers(2).Bias = avgParams(i).(['globalb', num2str(1)]);
            layers(5).Weights = avgParams(i).(['globalw', num2str(2)]);
            layers(5).Bias = avgParams(i).(['globalb', num2str(2)]);
            layers(8).Weights = avgParams(i).(['globalw', num2str(3)]);
            layers(8).Bias = avgParams(i).(['globalb', num2str(3)]);
            layers(11).Weights = avgParams(i).(['globalw', num2str(4)]);
            layers(11).Bias = avgParams(i).(['globalb', num2str(4)]);
            layers(13).Weights = avgParams(i).(['globalw', num2str(5)]);
            layers(13).Bias = avgParams(i).(['globalb', num2str(5)]);
        end

    % Loop to train network for each user...
    for user = 1:usernumber
        imdss = eval(sprintf('imds%d', user)); % Selecting dataset for current user
        % Train the network
        [netvaluable, info] = trainNetwork(imdss, layers, options);

    end

        % Calculate identification accuracy for the iteration using the test dataset
        predictedLabels = classify(netvaluable, imds_test);
        accuracy = sum(predictedLabels == imds_test.Labels) / numel(imds_test.Labels);
        accuracies(i) = accuracy * 100; % Store accuracy in percentage
        
        fprintf('Accuracy for iteration %d: %.2f%%\n', i, accuracies(i));
        
        % Update the plot with the new accuracy value
        plot(1:i, accuracies(1:i), '-o', 'LineWidth', 2);
        drawnow; % Ensure the plot updates in real-time
    end
    hold off; %release the figure
end