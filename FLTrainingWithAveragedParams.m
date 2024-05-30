
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    usernumber = 8;  
    varSize = 32; % Image dimension
    iterations = 70;
    repetitions = 1;
    ruche=true;
    % Initialize a figure for plotting accuracy
    % figure;
    % hold on;
    % xlabel('Iteration');
    % ylabel('Accuracy (%)');
    % title('Real-Time Accuracy Plot');
    % grid on;

    if ruche 
        refModelName = 'Ref_Model_70_i_3_r_noQ';
        directory_baseDir = 'refModel_70_i_3_avg';  
        directory_tempDir = sprintf('i_%d_avg_%d_AVGREF', iterations, repetitions);
        fullpath_baseDir = fullfile('/gpfs/workdir/costafrelu/RefModelParam_noQ_sameDS/', directory_baseDir);
        fullpath_tempDir = fullfile('/gpfs/workdir/costafrelu/temporaryMat/', directory_tempDir);
        if ~exist(fullpath_tempDir, 'dir')
            mkdir(fullpath_tempDir);
        end
    else
        refModelName = 'Ref_Model_1_i_1_r_noQ';
        directory_baseDir = 'refModel_1_i_1';
        directory_tempDir = sprintf('i_%d_avg_%d_AVGREF', iterations, repetitions);
        fullpath_baseDir = fullfile('..\workdir\RefModelParam_noQ_sameDS\', directory_baseDir);
        fullpath_tempDir = fullfile('..\workdir\temporaryMat\', directory_tempDir);
        if ~exist(fullpath_tempDir, 'dir')
            mkdir(fullpath_tempDir);
        end
    end
    
    % if isfile(fullpath_baseDir)
        load(fullfile(fullpath_baseDir, refModelName)); %carga allParams
        %load(filename, 'avgParams');
    % else
    %     error('Averaged parameters file %s not found.', fullpath_baseDir);
    % end

    loaded_imds1 = load(fullfile(fullpath_baseDir, 'imds1.mat'));
    loaded_imds2 = load(fullfile(fullpath_baseDir, 'imds2.mat'));
    loaded_imds3 = load(fullfile(fullpath_baseDir, 'imds3.mat'));
    loaded_imds4 = load(fullfile(fullpath_baseDir, 'imds4.mat'));
    loaded_imds5 = load(fullfile(fullpath_baseDir, 'imds5.mat'));
    loaded_imds6 = load(fullfile(fullpath_baseDir, 'imds6.mat'));
    loaded_imds7 = load(fullfile(fullpath_baseDir, 'imds7.mat'));
    loaded_imds8 = load(fullfile(fullpath_baseDir, 'imds8.mat'));
    imds1 = loaded_imds1.imds;
    imds2 = loaded_imds2.imds;
    imds3 = loaded_imds3.imds;
    imds4 = loaded_imds4.imds;
    imds5 = loaded_imds5.imds;
    imds6 = loaded_imds6.imds;
    imds7 = loaded_imds7.imds;
    imds8 = loaded_imds8.imds;
    
    loaded_imds_test = load(fullfile(fullpath_baseDir, 'imds_test.mat'));
    imds_test = loaded_imds_test.imds_test;
    
    
  accuracies = zeros(1, iterations);

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
            fullyConnectedLayer(10,'BiasLearnRateFactor',2);
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
        % plot(1:i, accuracies(1:i), '-o', 'LineWidth', 2);
        % drawnow; % Ensure the plot updates in real-time
    end

    save(fullfile(fullpath_tempDir, 'AccuracyAvgRefModel.mat'), 'accuracies'); 

    % hold off; %release the figure
