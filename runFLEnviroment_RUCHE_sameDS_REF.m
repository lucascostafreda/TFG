% 
% function [deviationFitness, accuracyFitness, RB_usedAverage, average_deviation, varianzas]  = runFLEnviroment_RUCHE_noQ_sameDS_2(PI, iteration, averagenumber, r,...
%     ruche,verbose, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, refModelName, directory_RefModel,directory_tempDir, FragSDS, percentages, iPI)
  
<<<<<<< HEAD
function [accuracyFitness]  = runFLEnviroment_RUCHE_sameDS_REF(iteration, averagenumber,...
    verbose, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, directory_RefModel, FragSDS, percentages, fullpath_tempDir, fullpath_baseDir2) 
=======
function [accuracyFitness]  = runFLEnviroment_RUCHE_sameDS_REF(iteration, averagenumber, ruche,...
    verbose, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, FragSDS, percentages, fullpath_tempDir, fullpath_baseDir2) 

>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
   % Este cambia en que imporatas la distribución del DS creado por el
   % RefModel
   % Además, hay que tener MUCHO cuidado que lo compares también con el REF
   % MOD que corresponde a la distribución anterior
    % Inicializar baseDir y tempDir según el valor de ruche 
    
<<<<<<< HEAD
     fullpath_baseDir = fullfile('/gpfs/workdir/costafrelu/RefModelParam_noQ_sameDS/', directory_RefModel);

    % load(fullfile(fullpath_baseDir, refModelName)); % Carga `allParams`

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
    
=======
     % fullpath_baseDir = fullfile('/gpfs/workdir/costafrelu/RefModelParam_noQ_sameDS/', directory_RefModel);

    % load(fullfile(fullpath_baseDir, refModelName)); % Carga `allParams`

    % loaded_imds1 = load(fullfile(fullpath_baseDir, 'imds1.mat'));
    % loaded_imds2 = load(fullfile(fullpath_baseDir, 'imds2.mat'));
    % loaded_imds3 = load(fullfile(fullpath_baseDir, 'imds3.mat'));
    % loaded_imds4 = load(fullfile(fullpath_baseDir, 'imds4.mat'));
    % loaded_imds5 = load(fullfile(fullpath_baseDir, 'imds5.mat'));
    % loaded_imds6 = load(fullfile(fullpath_baseDir, 'imds6.mat'));
    % loaded_imds7 = load(fullfile(fullpath_baseDir, 'imds7.mat'));
    % loaded_imds8 = load(fullfile(fullpath_baseDir, 'imds8.mat'));
    % imds1 = loaded_imds1.imds;
    % imds2 = loaded_imds2.imds;
    % imds3 = loaded_imds3.imds;
    % imds4 = loaded_imds4.imds;
    % imds5 = loaded_imds5.imds;
    % imds6 = loaded_imds6.imds;
    % imds7 = loaded_imds7.imds;
    % imds8 = loaded_imds8.imds;
    % 
    % loaded_imds_test = load(fullfile(fullpath_baseDir, 'imds_test.mat'));
    % imds_test = loaded_imds_test.imds_test;

>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    usernumber = 8;  
    varSize=32;
    
    %%%%%%%%%%%%%%%%%%%%%%%% coding setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     varSize = 32; 
    v_fQRate = [1, 2];
    v_nQuantizaers   = [...          % Curves
        0 ...                   % Dithered 3-D lattice quantization 
        1 ...                   % Dithered 2-D lattice quantization    
        1 ...                   % Dithered scalar quantization      
        1 ...                   % QSGD 
        1 ...                   % Uniform quantization with random unitary rotation    
        1 ...                   % Subsampling with 3 bits quantizers
        ];
    
    global gm_fGenMat2D;
    global gm_fLattice2D;
    % Clear lattices
    gm_fGenMat2D = [];
    gm_fLattice2D = [];
    % Do full search over the lattice
    stSettings.OptSearch = 1;
    
    stSettings.type =2;
    stSettings.scale=1;
    s_fRate=4;
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

    % deviation = zeros(iteration,averagenumber); 
    accuracy = zeros(iteration,averagenumber);
    accuracy2 = zeros(iteration,averagenumber);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % for rep = 1:1:PI_trials  
    for average=1:1:averagenumber
        
<<<<<<< HEAD
=======
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
        filename_acc = sprintf('Temp_Accuracy.mat');

        filename_params = sprintf('Ref_Model_%d.mat', average); % Auxiliar
        allParams = struct();

<<<<<<< HEAD
=======
    

>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
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
        fullyConnectedLayer(10,'BiasLearnRateFactor',2); %aqui he sacado length(categories)
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
        'Verbose', verbose);%
        
    for i=1:1:iteration
    
    % currentRefParams = avgParams(i);

<<<<<<< HEAD
    for user = 1:usernumber
        if FragSDS==1 && percentages(user)~=1
            % Load dataset for the user
            loaded_imds = load(fullfile(fullpath_baseDir, sprintf('imds%d.mat', user)));
            imds = loaded_imds.imds;
    
            % Shuffle and split the dataset according to the current percentage
            imds = shuffle(imds);
            imds = splitEachLabel(imds, percentages(user));  % Use the dynamically chosen percentage
    
            % Assign the processed data back to the original variable dynamically
            eval(sprintf('imds%d = imds;', user));
       end
    end

    for user=1:1:usernumber  

=======
    % for user = 1:usernumber
    %     if FragSDS==1 && percentages(user)~=1
    %         % Load dataset for the user
    %         loaded_imds = load(fullfile(fullpath_baseDir, sprintf('imds%d.mat', user)));
    %         imds = loaded_imds.imds;
    % 
    %         % Shuffle and split the dataset according to the current percentage
    %         imds = shuffle(imds);
    %         imds = splitEachLabel(imds, percentages(user));  % Use the dynamically chosen percentage
    % 
    %         % Assign the processed data back to the original variable dynamically
    %         eval(sprintf('imds%d = imds;', user));
    %    end
    % end

    for user=1:1:usernumber  

        imds = shuffle(imds);
        [splits{1:8}] = splitEachLabel(imds, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125);  % Split into 8 parts
        imds_user = splits{user}; 
        if FragSDS==1 && percentages(user)~=1
            imds_user = splitEachLabel(imds_user, percentages(user));
        end
        eval(sprintf('imds%d = imds_user;', user)); 

>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
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
    
    fprintf('Valor de user: %d, Valor de i: %d \n', user, i);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% User selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
        for user=1:1:usernumber
            deviationw1=deviationw1all(:,:,:,:,user);
            deviationw2=deviationw2all(:,:,:,:,user);
            deviationw3=deviationw3all(:,:,:,:,user);
            deviationw4=deviationw4all(:,:,user);
            deviationw5=deviationw5all(:,:,user);
            
            deviationb1=deviationb1all(:,:,:,user);
            deviationb2=deviationb2all(:,:,:,user);
            deviationb3=deviationb3all(:,:,:,user);
            deviationb4=deviationb4all(:,:,user);
            deviationb5=deviationb5all(:,:,user);   
                
            w1vector=reshape(deviationw1,[w1length,1]);
            w2vector=reshape(deviationw2,[w2length,1]);
            w3vector=reshape(deviationw3,[w3length,1]);
            w4vector=reshape(deviationw4,[w4length,1]);
            w5vector=reshape(deviationw5,[w5length,1]);   
            b1vector=reshape(deviationb1,[b1length,1]);
            b2vector=reshape(deviationb2,[b2length,1]);
            b3vector=reshape(deviationb3,[b3length,1]);
        
        
            m_fH1 = [w1vector;w2vector;w3vector;w4vector;w5vector;...
                    b1vector;b2vector;b3vector;deviationb4;deviationb5]; 
               
            [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings); % coding and decoding
            % bastante cambio aquí
            bstart=w1length+w2length+w3length+w4length+w5length;
           
         %%%%%%%%%%%%%%%% reshape the gradient of the loss function after coding %%%%%%%%%%%%  
            deviationw1=reshape(m_fHhat1(1:w1length),[5,5,3,32]);
            deviationw2=reshape(m_fHhat1(w1length+1:w1length+w2length),[5,5,32,32]);
            deviationw3=reshape(m_fHhat1(w1length+w2length+1:w1length+w2length+w3length),[5,5,32,64]);
            deviationw4=reshape(m_fHhat1(w1length+w2length+w3length+1:w1length+w2length+w3length+w4length),[64,576]);
            deviationw5=reshape(m_fHhat1(w1length+w2length+w3length+w4length+1:bstart),[10,64]);
            
            deviationb1(1,1,:)=reshape(m_fHhat1(bstart+1:bstart+b1length),[1,1,32]);
            deviationb2(1,1,:)=reshape(m_fHhat1(bstart+b1length+1:bstart+b1length+b2length),[1,1,32]);
            deviationb3(1,1,:)=reshape(m_fHhat1(bstart+b1length+b2length+1:bstart+b1length+b2length+b3length),[1,1,64]);
            deviationb4(:,1)=m_fHhat1(bstart+b1length+b2length+b3length+1:bstart+b1length+b2length+b3length+b4length);
            deviationb5(:,1)=m_fHhat1(bstart+b1length+b2length+b3length+b4length+1:bstart+b1length+b2length+b3length+b4length+b5length);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         %%%%%%%%%%%%%%%% calculate the local FL model of each user AFTER CODING %%%%%%%%%%%%  
         % esto se hace estrictamente para calcular el global 
         % (has codificado y decodificado los parametros)

         % global, la primera vez que se entra al bucle es = 0

            w1(:,:,:,:,user) = deviationw1 + globalw1;
            w2(:,:,:,:,user) = deviationw2 + globalw2;
            w3(:,:,:,:,user) = deviationw3 + globalw3;
            w4(:,:,user) = deviationw4 + globalw4;
            w5(:,:,user) = deviationw5 + globalw5; 
            
            b1(:,:,:,user) = deviationb1 + globalb1;
            b2(:,:,:,user) = deviationb2 + globalb2;
            b3(:,:,:,user) = deviationb3 + globalb3;
            b4(:,:,user) = deviationb4 + globalb4;
            b5(:,:,user) = deviationb5 + globalb5;
           
         end
            
        % Solo 1 vez por iteracion
         %%%%%%%%%%%%%%%% update the global FL model  %%%%%%%%%%%%  
        
            globalw1 = 1/length(user) * sum(w1(:,:,:,:,user), 5);  % global training model
            globalw2 = 1/length(user) * sum(w2(:,:,:,:,user), 5);  % global training model
            globalw3 = 1/length(user) * sum(w3(:,:,:,:,user), 5);
            globalw4 = 1/length(user) * sum(w4(:,:,user), 3);  
            globalw5 = 1/length(user) * sum(w5(:,:,user), 3);
        
            globalb1 = 1/length(user) * sum(b1(:,:,:,user), 4);  
            globalb2 = 1/length(user) * sum(b2(:,:,:,user), 4); 
            globalb3 = 1/length(user) * sum(b3(:,:,:,user), 4);
            globalb4 = 1/length(user) * sum(b4(:,:,user), 3);
            globalb5 = 1/length(user) * sum(b5(:,:,user), 3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%% UPDATE PI POLICY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
    
            % deviation(i,average) = objectiveFunction(globalw1, globalw2, globalw3, globalw4, globalw5, globalb1, globalb2, globalb3, globalb4, globalb5, currentRefParams);
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


    end
    
    save(fullfile(fullpath_tempDir,filename_params), 'allParams');

    %%%%%%%%%%%%%%%%%%%%%%%%% Iteration END %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end %end average

     AveragedRefModel(averagenumber,'Ref_Model_%d.mat',iteration, fullpath_baseDir2, fullpath_tempDir); 
     average_accuracy = zeros(iteration,1);     

    for ite=1:1:iteration
<<<<<<< HEAD
=======
        acc_aux=accuracy2(ite,:);
        acc_name= sprintf('AccuracyRefModel_%d',ite);
        save(fullfile(fullpath_tempDir, acc_name), 'acc_aux');
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
        average_accuracy(ite) = mean(squeeze(accuracy2(ite,:)), 'all');
    end
    
    %This is if later I would like to represent them.
    if AccDevMat
        % Guardar los archivos en las ubicaciones correspondientes
        save(fullfile(fullpath_tempDir, filename_acc), 'average_accuracy');
    end

    % Average resources used for each PI config
    accuracyFitness=average_accuracy(iteration);

end
