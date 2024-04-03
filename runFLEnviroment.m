function fitness = runFLEnviroment(PI, iteration, averagenumber, r)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% data processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    categories = {'deer','dog','frog','cat','bird','automobile','horse','ship','truck','airplane'};
    
    % rootFolder = '/gpfs/workdir/costafrelu/cifar10Test';
    rootFolder = 'cifar10Test';
    imds_test = imageDatastore(fullfile(rootFolder, categories), ...
        'LabelSource', 'foldernames');
    
    
     categories = {'deer','dog','frog','cat','bird','automobile','horse','ship','truck','airplane'};
    
    % rootFolder = '/gpfs/workdir/costafrelu/cifar10Train';
    rootFolder = 'cifar10Train';
    imds = imageDatastore(fullfile(rootFolder, categories), ... 
        'LabelSource', 'foldernames');

     %%
    %%%%%%%%%%%%%%%%%%%% IID dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    [imds1, imds2, imds3, imds4, imds5, imds6, imds7, imds8] = splitEachLabel(imds, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125);
    
    % load('/gpfs/workdir/costafrelu/Ref_Model_65_i_6_r.mat'); % Loads `allParams`
    load('Ref_Model_65_i_6_r.mat'); % Loads `allParams`
    
    % averagenumber = 1;  % Average number of runing simulations.  Repeticiones, para promediar el error del pack al final !!
    % iteration = 5;     % Total number of global FL iterations.
    usernumber = 8;
    
    % total_available_RBs = 450; % 75% of the total required to transmit them all at once
    
    % Static definition of CQI
    % CQI_indices = randi([1 15], 1, usernumber);
    % CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
    % [r, eta, sigma] = parameters(CQI_indices);
    
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PI_trials=1;
    average_RB = zeros(PI_trials,1);
    RB_used = zeros(PI_trials,averagenumber,iteration);
    
    deviation = zeros(iteration,averagenumber); 
    accuracy = zeros(iteration,averagenumber);
    accuracy2 = zeros(iteration,averagenumber);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for rep = 1:1:PI_trials
        
    for average=1:1:averagenumber

    % Concatenate to form the new vector [a1, a2, x1, x2, a3, a4, x1, x2]
    % b1=[a1,a2];
    % b2=[a3,a4];
    % PI = [b1, updated_PI, b2, updated_PI];  % Use semi-colon for column-wise concatenation

        filename_dev = sprintf('deviation_PI_%d.mat', rep);
        filename_acc = sprintf('Accuracy_PI_%d.mat', rep);
    
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%% gradient of local FL models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    deviationw=[];
    deviationlw=[];
    deviationb=[];
    deviationob=[];
    deviationofw=[];
    
    %%%%%%%%%%  Building local FL model of each user  %%%%%%%%%%%%%%%%%%%%%%%%%
    
    
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
        
    for i=1:1:iteration
    
    %currentRefParams = allParams.(['iter_' num2str(i)]);
    %currentRefParams = allParams(i);
    currentRefParams = avgParams(i);
    
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
    
    % Proceed with training if the subset is valid
    [netvaluable, info] = trainNetwork(imdss, layer, option);
    
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
    
    fprintf('Valor de user: %d, Valor de i: %d\n', user, i);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% User selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % selecciono según distribución de los recursos
    % OJO, PI aquí está totalmente comprometida a r. 
    % Ahora será PI la que cambie iterativamente
    
            % available_RBs = total_available_RBs;
            [selected_users, total_resource_usage] = selectUsersBasedOnPI(PI, r, usernumber);
            RB_used(rep,average,i) = total_resource_usage;
            
            fprintf('Iteration %d: Selected users %s\n', i, mat2str(selected_users));
    
    % ENVÍO PARAM.
    % Si no, pasa directamente a la siguiente iteración
    
        % Empieza todo el proceso para actualizar el modelo global
        % 1. Procesar las desviaciones
        % 2. Codificarlas y decodificarlas
        % 3. Usar estas para actualizar el modelo global
    
            for idx=1:1:length(selected_users)
            
                user = selected_users(idx);
    
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
        
            globalw1 = 1/length(selected_users) * sum(w1(:,:,:,:,selected_users), 5);  % global training model
            globalw2 = 1/length(selected_users) * sum(w2(:,:,:,:,selected_users), 5);  % global training model
            globalw3 = 1/length(selected_users) * sum(w3(:,:,:,:,selected_users), 5);
            globalw4 = 1/length(selected_users) * sum(w4(:,:,selected_users), 3);  
            globalw5 = 1/length(selected_users) * sum(w5(:,:,selected_users), 3);
        
            globalb1 = 1/length(selected_users) * sum(b1(:,:,:,selected_users), 4);  
            globalb2 = 1/length(selected_users) * sum(b2(:,:,:,selected_users), 4); 
            globalb3 = 1/length(selected_users) * sum(b3(:,:,:,selected_users), 4);
            globalb4 = 1/length(selected_users) * sum(b4(:,:,selected_users), 3);
            globalb5 = 1/length(selected_users) * sum(b5(:,:,selected_users), 3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%% UPDATE PI POLICY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
    
            deviation(i,average) = objectiveFunction(globalw1, globalw2, globalw3, globalw4, globalw5, globalb1, globalb2, globalb3, globalb4, globalb5, currentRefParams);
    
    %%%%%%%%%%%%%%%%%%%%%%%%% PRINT RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    end
    
    end %end average
    
     average_deviation = zeros(iteration,1);
     average_accuracy = zeros(iteration,1);
    
    for ite=1:1:iteration
        average_deviation(ite) = mean(squeeze(deviation(ite,:)), 'all');
        average_accuracy(ite) = mean(squeeze(accuracy2(ite,:)), 'all');
    end
    
    %filename_dev = fullfile('/gpfs/workdir/costafrelu/', filename_dev);
       filename_dev = fullfile('C:\Users\Lucas\OneDrive\Documentos\MATLAB\TFG\workdir\', filename_dev);
    save(filename_dev, 'average_deviation');
    %filename_acc = fullfile('/gpfs/workdir/costafrelu/', filename_acc);
    filename_acc = fullfile('C:\Users\Lucas\OneDrive\Documentos\MATLAB\TFG\workdir\', filename_acc);
    save(filename_acc, 'average_accuracy'); % de momento solo cogerá la de la segunda average
    
    %save('gradient_vector.mat', 'totalGradient');

    %Average resources used for each PI config
    RB_usedAverage(rep) =  mean(squeeze(RB_used(rep,:,:)), 'all');
    
    end
    
    for rep=1:1:PI_trials
            fprintf('Rep %d, Average RB = %.4f\n', rep, RB_usedAverage(rep));
    end
    %comparePIConfigurations();
    fitness=average_deviation(iteration); %Instead of averaging, we get the last one
    
end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [selected_users, total_resource_usage] = selectUsersBasedOnPI(PI, r, num_users)
    %The restriction is already met sum(PI.*r)<=R
    selected_users = [];
    total_resource_usage = 0;
    % Independently select users based on PI, ensuring resource constraints are respected
    for u = 1:num_users
        if rand() <= PI(u)
            selected_users = [selected_users, u];
            total_resource_usage = total_resource_usage + r(u);
        end
    end
    
end

function deviation = objectiveFunction(globalw1, globalw2, globalw3, globalw4, globalw5, globalb1, globalb2, globalb3, globalb4, globalb5, currentRefParams)
    
    % Calculate individual squared differences
    diff_globalw1 = sum(abs(globalw1(:) - currentRefParams.globalw1(:)).^2, 'all');
    diff_globalw2 = sum(abs(globalw2(:) - currentRefParams.globalw2(:)).^2, 'all');
    diff_globalw3 = sum(abs(globalw3(:) - currentRefParams.globalw3(:)).^2, 'all');
    diff_globalw4 = sum(abs(globalw4(:) - currentRefParams.globalw4(:)).^2, 'all');
    diff_globalw5 = sum(abs(globalw5(:) - currentRefParams.globalw5(:)).^2, 'all');
    diff_globalb1 = sum(abs(globalb1(:) - currentRefParams.globalb1(:)).^2, 'all');
    diff_globalb2 = sum(abs(globalb2(:) - currentRefParams.globalb2(:)).^2, 'all');
    diff_globalb3 = sum(abs(globalb3(:) - currentRefParams.globalb3(:)).^2, 'all');
    diff_globalb4 = sum(abs(globalb4(:) - currentRefParams.globalb4(:)).^2, 'all');
    diff_globalb5 = sum(abs(globalb5(:) - currentRefParams.globalb5(:)).^2, 'all');
   
    total_deviation = diff_globalw1 + diff_globalw2 + diff_globalw3 + diff_globalw4 + diff_globalw5 + ...
                      diff_globalb1 + diff_globalb2 + diff_globalb3 + diff_globalb4 + diff_globalb5;
    
    deviation = sqrt(total_deviation);

end

% Constraints Function
function [c, ceq] = constraints(PI, r, total_available_RBs)
    c = sum(PI .* r) - total_available_RBs; % Resource constraint
    ceq = []; % No equality constraints
end
