    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% data processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ruche = true; %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
categories = {'deer','dog','frog','cat','bird','automobile','horse','ship','truck','airplane'};
    
    if ruche
        rootFolder = '/gpfs/workdir/costafrelu/cifar10Test';
    else
        rootFolder = 'cifar10Test';
    end
     
    imds_test = imageDatastore(fullfile(rootFolder, categories), ...
        'LabelSource', 'foldernames');
    
    
     categories = {'deer','dog','frog','cat','bird','automobile','horse','ship','truck','airplane'};
    
     if ruche
         rootFolder = '/gpfs/workdir/costafrelu/cifar10Train';
     else 
         rootFolder = 'cifar10Train';
     end
      
    imds = imageDatastore(fullfile(rootFolder, categories), ... 
        'LabelSource', 'foldernames');

    usernumber = 8;
    executionEnviroment = 'parallel';
    miniBatchSize = 100;
    verbose = true;
    averagenumber = 3;
    iteration = 70;
    varSize = 32;
    Shuffle='never';

    %%%%%%%%%%%%%%%%%%%% IID dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   %%
    directory = sprintf('refModel_%d_i_%d_avg', iteration, averagenumber);

    if ruche
        baseDir = '/gpfs/workdir/costafrelu/RefModelParam_noQ_sameDS/';
        tempDir = '/gpfs/workdir/costafrelu/temporaryMat/';
        fullPath_baseDir = fullfile(baseDir, directory);
        fullPath_tempDir = fullfile(tempDir, directory);
    else
        baseDir = '..\workdir\RefModelParam_noQ_sameDS';
        tempDir = '..\workdir\temporaryMat';
        fullPath_baseDir = fullfile(baseDir, directory);
        fullPath_tempDir = fullfile(tempDir, directory);
    end
    
    if ~exist(fullPath_baseDir, 'dir') || ~exist(fullPath_tempDir, 'dir')
        mkdir(fullPath_baseDir);
        mkdir(fullPath_tempDir);
    end
    
%%
    [imds1, imds2, imds3, imds4, imds5, imds6, imds7, imds8] = splitEachLabel(imds, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125);

    
    % Percentages for the second split
    percentages = [1/1, 3/4, 1/2, 1/4, 1/8, 1/16];
    
    imds1 = shuffle(imds1);
    imds2 = shuffle(imds2);
    imds3 = shuffle(imds3);
    imds4 = shuffle(imds4);
    imds5 = shuffle(imds5);
    imds6 = shuffle(imds6);
    imds7 = shuffle(imds7);
    imds8 = shuffle(imds8);

    for i = 1:8
    % Dynamically load the dataset
    imds = eval(sprintf('imds%d', i));
    
    % Save the shuffled datasets
    save(fullfile(fullPath_baseDir, sprintf('imds%d.mat', i)), 'imds');
        
    % Loop through each percentage to split the dataset and save each split
    for j = 1:length(percentages)
        % Split the dataset based on the current percentage
        imds_percentSplit = splitEachLabel(imds, percentages(j));
        % Save the split dataset, including the dataset index and the percentage index
        save(fullfile(fullPath_baseDir, sprintf('imds%d_percentSplit%d.mat', i, j)), 'imds_percentSplit');
    end
    end

   save(fullfile(fullPath_baseDir, 'imds_test.mat'),'imds_test');

    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Matrix size of local FL model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    w1length=5*5*3*32;
    w2length=5*5*32*32;
    w3length=5*5*32*64;
    w4length=64*576;
    w5length=10*64;
    
    b1length=32;
    b2length=32;
    b3length=64;
    b4length=64;
    b5length=10;
    %%%%%%%%%%%%% Deviations, Resources and Accuracy VECTORS %%%%%%%%%%%%%%%
    
    RB_used = zeros(averagenumber,iteration);

    deviation = zeros(iteration,averagenumber); 
    accuracy = zeros(iteration,averagenumber);
    accuracy2 = zeros(iteration,averagenumber);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for average=1:1:averagenumber

    % Nombramiento files
    filename_params = sprintf('Ref_Model_%d.mat', average); % Auxiliar
    filename_acc=sprintf('accuracy_refModel_%d_i_%d_av.mat', iteration, averagenumber);

    allParams = struct();
    
    %%%%%%%%%%%%% local model of each user%%%%%%%%%%%%%%%%%%%%%%%  
    w1=[];
    w2=[];
    w3=[];
    w4=[];
    w5=[];
    b1=[];
    b2=[];
    b3=[];
    b4=[];
    b5=[];

    %%%%%%%%%%%%  Building local FL model of each user  %%%%%%%%%%%%%%%%%%%%

    layer = [
        imageInputLayer([varSize varSize 3]); % input layer with the size of the input images
        convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2); % 2D convolutional layer with 5x5 filters.
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
        'MiniBatchSize', miniBatchSize, ... % Batch Entero %CHANGE
        'ExecutionEnvironment', executionEnviroment,... %cannot be parallel-auto
        'Shuffle', Shuffle,... % mezclar antes de cada epoch, no sirve de mucho, pues pasasuna vez
        'Verbose', verbose); %
        %'WorkerLoad', ones(1, 4)); % Ajusta basado en pruebas para tu caso específico
        %'Plots', 'training-progress');

        % Initialize the figure for plotting

    fig1=figure;
    hold on; 
    
    xlabel('Iteration');
    ylabel('Current Accuracy');
    title('Real-Time Accuracy Plot (RM_NE)');
    grid on;
 
    for i=1:1:iteration
        
    for user=1:1:usernumber  
        
        clear netvaluable;
        Winstr1=strcat('net',int2str(user));     
        midstr=strcat('imds',int2str(user)); 
        eval(['imdss','=',midstr,';']);
    
    if i>1   
        % ite después de haber actualizado el modelo global (que no es la
        % primera)
    
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
    % tic
    % Proceed with training if the subset is valid
    [netvaluable, info] = trainNetwork(imdss, layer, option);
    
    % tiempo=toc;
    %%%%%%%%%%%%%%%%%%% calculate identification accuracy %%%%%%%%%%%%%%%%%%%%%
    
    labels = classify(netvaluable, imds_test);
    
    confMat = confusionmat(imds_test.Labels, labels);
    confMat = confMat./sum(confMat,2);
    accuracy(i,average)=mean(diag(confMat))+accuracy(i,average); 
    accuracy2(i,average) = (accuracy(i,average)/usernumber)*100;
    
    
    %%%%%%%%%%%%% global model for each user, which consists of 4 matrices  
    
    if i==1    
        globalw1=zeros(5,5,3,32);
        globalw2=zeros(5,5,32,32);
        globalw3=zeros(5,5,32,64);
        globalw4=zeros(64,576);
        globalw5=zeros(10,64);
        
        globalb1=zeros(1,1,32);
        globalb2=zeros(1,1,32);
        globalb3=zeros(1,1,64);
        globalb4=zeros(64,1);
        globalb5=zeros(10,1);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Record trained local FL model.
    % registramos pesos para operar con ellos
    
        w1(:,:,:,:,user)=netvaluable.Layers(2).Weights;
        w2(:,:,:,:,user)=netvaluable.Layers(5).Weights;
        w3(:,:,:,:,user)=netvaluable.Layers(8).Weights;
        w4(:,:,user)=netvaluable.Layers(11).Weights;
        w5(:,:,user)=netvaluable.Layers(13).Weights;
            
        b1(:,:,:,user)=netvaluable.Layers(2).Bias;    
        b2(:,:,:,user)=netvaluable.Layers(5).Bias;
        b3(:,:,:,user)=netvaluable.Layers(8).Bias;
        b4(:,:,user)=netvaluable.Layers(11).Bias;
        b5(:,:,user)=netvaluable.Layers(13).Bias;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%% Calculate the gradient of local FL model of each user  %%%%%%%    
    
    % IMPORTANTÍSIMO, LOS GRADIENTES LOCALES SON NUEVOS CONSTANTEMENTE, POR LO QUE
    % SE VUELVEN A CALCULAR TENIENDO EN CUENTA LAS NUEVAS ACTUALIZACIONES DEL LOS PARAM LOCALES
    
        deviationw1all(:,:,:,:,user)= w1(:,:,:,:,user)-globalw1; 
        deviationw2all(:,:,:,:,user)=w2(:,:,:,:,user)-globalw2;
        deviationw3all(:,:,:,:,user)= w3(:,:,:,:,user)-globalw3;
        deviationw4all(:,:,user)=w4(:,:,user)-globalw4;
        deviationw5all(:,:,user)=w5(:,:,user)-globalw5;
        
        deviationb1all(:,:,:,user)=b1(:,:,:,user)-globalb1;
        deviationb2all(:,:,:,user)=b2(:,:,:,user)-globalb2;
        deviationb3all(:,:,:,user)=b3(:,:,:,user)-globalb3;
        deviationb4all(:,:,user)=b4(:,:,user)-globalb4;
        deviationb5all(:,:,user)= b5(:,:,user)-globalb5;    
            
    %end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        w1(:,:,:,:,user) = deviationw1all(:,:,:,:,user) + globalw1;
        w2(:,:,:,:,user) = deviationw2all(:,:,:,:,user) + globalw2;
        w3(:,:,:,:,user) = deviationw3all(:,:,:,:,user) + globalw3;
        w4(:,:,user) = deviationw4all(:,:,user) + globalw4;
        w5(:,:,user) = deviationw5all(:,:,user) + globalw5;
        
        b1(:,:,:,user) = deviationb1all(:,:,:,user) + globalb1;
        b2(:,:,:,user) = deviationb2all(:,:,:,user) + globalb2;
        b3(:,:,:,user) = deviationb3all(:,:,:,user) + globalb3;
        b4(:,:,user) = deviationb4all(:,:,user) + globalb4;
        b5(:,:,user) = deviationb5all(:,:,user) + globalb5;

    fprintf('User: %d, Valor de i: %d \n', user, i);
    end
        
     %%%%%%%%%%%%%%%% update the global FL model  %%%%%%%%%%%%  
    
        globalw1 = 1/usernumber * sum(w1, 5);  % global training model
        globalw2 = 1/usernumber * sum(w2, 5);  % global training model
        globalw3 = 1/usernumber * sum(w3, 5);
        globalw4 = 1/usernumber * sum(w4, 3);  
        globalw5 = 1/usernumber * sum(w5, 3);
    
        globalb1 = 1/usernumber * sum(b1, 4);  
        globalb2 = 1/usernumber * sum(b2, 4); 
        globalb3 = 1/usernumber * sum(b3, 4);
        globalb4 = 1/usernumber * sum(b4, 3);
        globalb5 = 1/usernumber * sum(b5, 3);
        

        allParams(i).globalw1 = globalw1;
        allParams(i).globalw2 = globalw2;
        allParams(i).globalw3 = globalw3;
        allParams(i).globalw4 = globalw4;
        allParams(i).globalw5 = globalw5;
        allParams(i).globalb1 = globalb1;
        allParams(i).globalb2 = globalb2;
        allParams(i).globalb3 = globalb3;
        allParams(i).globalb4 = globalb4;
        allParams(i).globalb5 = globalb5;


        figure(fig1);
        plot(i, accuracy2(i), 'bo-');
        drawnow;
    
    end

    % Define full file paths
    % filename_acc = fullfile(fullPath_baseDir, filename_acc);
    % filename_params = fullfile(fullPath_tempDir, filename_params);
    
    % Save files
    save(fullfile(fullPath_baseDir, filename_acc), 'accuracy2');
    save(fullfile(fullPath_tempDir, filename_params), 'allParams');

    %%%%%%%%%%%%%%%%%%%%%%%%% Iteration END %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end %end average
    
    ComputeModelVariability(averagenumber,'Ref_Model_%d.mat',iteration, fullPath_baseDir, fullPath_tempDir); 
    
 
    



