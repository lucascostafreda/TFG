<<<<<<< HEAD
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%% data processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
categories = {'deer','dog','frog','cat','bird','automobile','horse','ship','truck','airplane'};

rootFolder = 'cifar10Test';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');


 categories = {'deer','dog','frog','cat','bird','automobile','horse','ship','truck','airplane'};

rootFolder = 'cifar10Train';
imds = imageDatastore(fullfile(rootFolder, categories), ... 
    'LabelSource', 'foldernames');
 %%
%%%%%%%%%%%%%%%%%%%% IID dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
[imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10,imds11,imds12,imds13,imds14,imds15,imds16] = splitEachLabel(imds, 0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625);

R=40;
T=7;

indices=[12 10 8 6 12 10 8 6 12 10 8 6 12 10 8 6];

[r,eta,sigma]=parameters(indices);     
[PI,error,errormin,errorrandom]= Optimal_Sampling_Min_Error(r,eta,sigma,T,R);

% espcify a metric insted of the MMSE, related to performance of the FL
% compare with centralised 
% assumption for 1 ms, what task can be achieved
% limitation 1ms, every 1ms will make an update to the CC
% for each epoch x

% HOMOGENEOUS DATASETS
% Assuming PI is defined with values between 0 and 1 for each subset
%PI = [0.5, 0.75, 0.6, 0.8, 0.65, 0.7, 0.55, 0.6, 0.75, 0.5, 0.85, 0.8, 0.7, 0.75, 0.65, 0.6];

% Iterate over each subset and adjust its size based on the corresponding PI value
for i = 1:length(PI)
    % Fetch the current subset based on the loop index
    subsetName = sprintf('imds%d', i);
    imdsSubset = eval(subsetName);  % Get the current subset using dynamic variable names
    
    % Placeholder arrays for collecting the selected files and their labels
    adjustedFiles = {};
    adjustedLabels = [];
    
    % Retrieve the count of images per category in the current subset
    labelCounts = countEachLabel(imdsSubset); % already works in the categorical level
    uniqueLabels = unique(imdsSubset.Labels); % Get unique labels directly from the subset
    
    for label = 1:numel(uniqueLabels)
        % Indices of all images belonging to the current category
        categoryIndices = find(imdsSubset.Labels == uniqueLabels(label));

        % The target number of images for this category, rounded to the nearest integer
        newSize = round(numel(categoryIndices) * PI(i)); 

        % Randomly select 'targetCount' indices for this category
        selectedIndices = categoryIndices(randperm(numel(categoryIndices), newSize));
        
        % Append the selected files and labels for this category to the adjusted lists
        adjustedFiles = [adjustedFiles; imdsSubset.Files(selectedIndices)];
        adjustedLabels = [adjustedLabels; repmat(uniqueLabels(label), newSize, 1)];
    end
    
    % Create a new ImageDatastore for the adjusted subset
    AdjustedSubsets{i} = imageDatastore(adjustedFiles, 'Labels', categorical(adjustedLabels));
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

averagenumber=1;  % Average number of runing simulations.  Repeticiones, para promediar el error del pack al final !!
iteration=250;     % Total number of global FL iterations.
number=0; %to keep truck of 
usernumber = 15;
proposed=1;
alpha=0.85;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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


finalerror=[];
averageerror=[];
kk=0;



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

for userno=10:2:10    % Number of users.
    kk=kk+1; % cuantos packs de usuarios hago para después representarlo mejor
    numberofuser=userno; 
    
for average=1:1:averagenumber % cuantas veces lo hago para hacer un promediado

d=[407.3618
  452.8960
   63.4934
  456.6879
  316.1796
   48.7702
  139.2491
  273.4408
  478.7534
  482.4443
   78.8065
  485.2964
  478.5835
  242.6878
  400.1402]; 

error=zeros(iteration,1); % inicializa el error cometido en cada iteración
    
gradient=zeros(usernumber,1);
wupdate=zeros(iteration,usernumber);   % local model for each user

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

wnew=zeros(5,5,3,32,usernumber);
lwnew=zeros(5,5,32,32,usernumber);
bnew=zeros(5,5,32,64,usernumber);
obnew=zeros(64,576,usernumber);
fwnew=zeros(10,64,usernumber); %weights of the final fully connected layer 
% in the network, mapping 64 input features to 10 output classes



%%%%%%%%%%%%% gradient of local FL models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
deviationw=[];
deviationlw=[];
deviationb=[];
deviationob=[];
deviationofw=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    'Verbose', true);%,... 
    %'Plots', 'training-progress');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize the figure for plotting
figure;
hold on; % Keep the plot from being erased on new updates
xlabel('Iteration');
ylabel('Current Accuracy');
title('Real-Time Accuracy Plot');
grid on;

kkkk=0;
for i=1:1:iteration
    
%%%%%%%%%%%%%%%%%%%%%%%Setting of local FL model %%%%%%%%%%%%%%%%%%%%%%%%%%   
error2 = zeros(iteration, 1); % Pre-allocate para eficiencia

for user=1:1:usernumber  
    
    clear netvaluable;
    %Winstr1=strcat('net',int2str(user));     
    midstr = strcat('AdjustedSubsets{', int2str(user), '}');
    eval(['imdsPi =', midstr, ';']);

    


if i>1
   %the code updates the weights and biases every iteration of each user
   %with the global CC model, except the first one

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
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
      
    % Check if the current subset is empty or has no labels
    if isempty(imdsPi.Files)
        warning('Skipping training for user %d in iteration %d due to no data.', user, i);
        modelExists = false; % Flag indicating the absence of a model
    else
        % Proceed with training if the subset is valid
        [netvaluable, info] = trainNetwork(imdsPi, layer, option); % Train local FL model.
        modelExists = true;
    end
   % por red se refiere a una red neuronal entrenada, objeto Series Network

%%%%%%%%%%%%%%%%%%%calculate identification accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %labels = classify(netvaluable, imds_test);

if modelExists
    labels = classify(netvaluable, imds_test);

    confMat = confusionmat(imds_test.Labels, labels);
    confMat = confMat./sum(confMat,2);
    error(i,1)=mean(diag(confMat))+error(i,1); 

else
    warning('Model for user %d is empty. Skipping classification.',user);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% se vuelven a recolectar para ver más adelante cual es el más provechoso
% de los trainings x user
if modelExists

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
else
    warning('Model for user %d is empty. Skipping classification.',user);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     
    
%%%%%%%%%%%%% Calculate the gradient of local FL model of each user  %%%%%%%    

if modelExists 
    if i == 1 % no se considera GLOBAL porque aun no se ha  actualizado
        deviationw1all(:,:,:,:,user)= w1(:,:,:,:,user);
        deviationw2all(:,:,:,:,user)=w2(:,:,:,:,user);
        deviationw3all(:,:,:,:,user)= w3(:,:,:,:,user);
        deviationw4all(:,:,user)=w4(:,:,user);
        deviationw5all(:,:,user)=w5(:,:,user);
        
        deviationb1all(:,:,:,user)=b1(:,:,:,user);
        deviationb2all(:,:,:,user)=b2(:,:,:,user);
        deviationb3all(:,:,:,user)=b3(:,:,:,user);
        deviationb4all(:,:,user)=b4(:,:,user);
        deviationb5all(:,:,user)= b5(:,:,user);
    else
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
    end
else

    fprintf('Model not trained for user %d in the iteration %d.\n', user, i);
    
    % desviaciones a 0, pues Netvaluable no se ha entrenado
    deviationw1all(:,:,:,:,user) = zeros(size(w1, 1), size(w1, 2), size(w1, 3), size(w1, 4));
    deviationw2all(:,:,:,:,user) = zeros(size(w2, 1), size(w2, 2), size(w2, 3), size(w2, 4));
    deviationw3all(:,:,:,:,user) = zeros(size(w3, 1), size(w3, 2), size(w3, 3), size(w3, 4));
    deviationw4all(:,:,user) = zeros(size(w4, 1), size(w4, 2));
    deviationw5all(:,:,user) = zeros(size(w5, 1), size(w5, 2));
    
    deviationb1all(:,:,:,user) = zeros(size(b1, 1), size(b1, 2), size(b1, 3));
    deviationb2all(:,:,:,user) = zeros(size(b2, 1), size(b2, 2), size(b2, 3));
    deviationb3all(:,:,:,user) = zeros(size(b3, 1), size(b3, 2), size(b3, 3));
    deviationb4all(:,:,user) = zeros(size(b4, 1), size(b4, 2));
    deviationb5all(:,:,user) = zeros(size(b5, 1), size(b5, 2));
end

fprintf('Valor de user: %d, Valor de i: %d\n', user, i);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% user selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if proposed==1

for jjj=1:1:usernumber %all deviations x user
gradient(jjj,1)=sum(sqrt(deviationw1all(:,:,:,:,jjj).^2),'all')+sum(sqrt(deviationw2all(:,:,:,:,jjj).^2),'all')+sum(sqrt(deviationw3all(:,:,:,:,jjj).^2),'all')...
                +sum(sqrt(deviationw4all(:,:,jjj).^2),'all')+sum(sqrt(deviationw5all(:,:,jjj).^2),'all')+sum(sqrt(deviationb1all(:,:,:,jjj).^2),'all')...
                +sum(sqrt(deviationb2all(:,:,:,jjj).^2),'all')+sum(sqrt(deviationb3all(:,:,:,jjj).^2),'all')...
                +sum(sqrt(deviationb4all(:,:,jjj).^2),'all')+sum(sqrt(deviationb5all(:,:,jjj).^2),'all');

% suma total de todos los gradientes
% gradiente is a vector of Usernumber columns and 1 row
end
% algunos gradientes son 0

% OJO!! reconsiderar política para los usuarios escogidos (si hay muchos
% sin escoger, saber que solo he de usar los !=0)

 [xx,dex]=sort(gradient); % [sorted output (value), sort index]
 
    probability=gradient/sum(gradient); % prob por gradiente

    % if probability == 0
    
    probabilityd=(max(d)-d)/sum(max(d)-d); % prob por distancia
    probability=probability*alpha+(1-alpha)*probabilityd; % prob conjunta
    vector=[1:1:usernumber];
      
     %Here it only selects 10/15 users

     % actualizarnumberofusers

    for llll=1:1:numberofuser

        bb(i,llll)=randsrc(1,1,[vector; probability']); %inicializa bb, 
        % selecciona un elem de forma aleatoria (1,1) de vector.
        % las opciones de ser seleccionado vienen marcadas por probability'
        
        position=find(vector==bb(i,llll)); %finds index of the element in 'vect'
        % () that matches value stores in bb
        probability1=probability(position,1); % retreives prob associated
        probability(position)=[]; %removes the elem
        vector(position)=[]; % removes selected elem from vect
        probability=probability./(1-probability1);
        % maintain the probability distribution's integrity after an element 
        % has been removed.
        end
     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% reshape the gradient of local FL model of each user   %%%%%%%        

 
for code=1:1:numberofuser
% trabajar solo con las desviaciones de los modelos de usuarios que han sido 
% escogidos para contribuir en esta ronda de aprendizaje federado.

%MIRAR SI AQUÍ TB TENEMOS QUE PONER WARNINGS POR SI NO EXISTEN LAS DEV

deviationw1=deviationw1all(:,:,:,:,bb(i,code)); 
deviationw2=deviationw2all(:,:,:,:,bb(i,code));
deviationw3=deviationw3all(:,:,:,:,bb(i,code));
deviationw4=deviationw4all(:,:,bb(i,code));
deviationw5=deviationw5all(:,:,bb(i,code));

deviationb1=deviationb1all(:,:,:,bb(i,code));
deviationb2=deviationb2all(:,:,:,bb(i,code));
deviationb3=deviationb3all(:,:,:,bb(i,code));
deviationb4=deviationb4all(:,:,bb(i,code));
deviationb5=deviationb5all(:,:,bb(i,code));  
    
% matriz de desviaciones seleccionada se convierte en un vector.  
% preparar las desviaciones para su posterior procesamiento
    
w1vector=reshape(deviationw1,[w1length,1]);

w2vector=reshape(deviationw2,[w2length,1]);

w3vector=reshape(deviationw3,[w3length,1]);

w4vector=reshape(deviationw4,[w4length,1]);

w5vector=reshape(deviationw5,[w5length,1]);   


b1vector=reshape(deviationb1,[b1length,1]);

b2vector=reshape(deviationb2,[b2length,1]);

b3vector=reshape(deviationb3,[b3length,1]);
    %concatenan los vectores de desviaciones de pesos
    m_fH1 = [w1vector;w2vector;w3vector;w4vector;w5vector;...
            b1vector;b2vector;b3vector;deviationb4;deviationb5]; 
    
    [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings); % coding and decoding

    % índice de inicio para los sesgos
    bstart=w1length+w2length+w3length+w4length+w5length;
    
    %%%%%%%%%%%%%%%% reshape the gradient of the loss function after coding %%%%%%%%%%%%  
    % Reorganizados en sus formatos originales
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


 %%%%%%%%%%%%%%%% calculate the local FL model of each user after coding %%%%%%%%%%%%  

    if i==1
    % Se actualizan los nuevos pesos y sesgos de los usuarios escogidos.
    % Para poder hacer una media (de los 10 elegidos) y actualizar el modelo del CC global
        w1(:,:,:,:,bb(i,code))=deviationw1;
        w2(:,:,:,:,bb(i,code))=deviationw2;
        w3(:,:,:,:,bb(i,code))=deviationw3;
        w4(:,:,bb(i,code))=deviationw4;
        w5(:,:,bb(i,code))=deviationw5;
        
        b1(:,:,:,bb(i,code))=deviationb1;
        b2(:,:,:,bb(i,code))=deviationb2;
        b3(:,:,:,bb(i,code))=deviationb3;
        b4(:,:,bb(i,code))=deviationb4;
        b5(:,:,bb(i,code))=deviationb5;
          
    else       
        w1(:,:,:,:,bb(i,code))=deviationw1+globalw1;
        w2(:,:,:,:,bb(i,code))=deviationw2+globalw2;
        w3(:,:,:,:,bb(i,code))=deviationw3+globalw3;
        w4(:,:,bb(i,code))=deviationw4+globalw4;
        w5(:,:,bb(i,code))=deviationw5+globalw5;
        
        b1(:,:,:,bb(i,code))=deviationb1+globalb1;
        b2(:,:,:,bb(i,code))=deviationb2+globalb2;
        b3(:,:,:,bb(i,code))=deviationb3+globalb3;
        b4(:,:,bb(i,code))=deviationb4+globalb4;
        b5(:,:,bb(i,code))=deviationb5+globalb5;     
    
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
 %%%%%%%%%%%%%%%% update the global FL model  %%%%%%%%%%%% 

 % Finalmente se actualiza el modelo global
 % Al principio del bucle (iteracion) se actualiza cada usuario con este
 % modelo global.

globalw1=1/numberofuser*sum(w1(:,:,:,:,bb(i,:)),5);  % global training model
globalw2=1/numberofuser*sum(w2(:,:,:,:,bb(i,:)),5);  % global training model
globalw3=1/numberofuser*sum(w3(:,:,:,:,bb(i,:)),5);
globalw4=1/numberofuser*sum(w4(:,:,bb(i,:)),3);

globalw5=1/numberofuser*sum(w5(:,:,bb(i,:)),3);

globalb1=1/numberofuser*sum(b1(:,:,:,bb(i,:)),4);  % global training model
globalb2=1/numberofuser*sum(b2(:,:,:,bb(i,:)),4);  % global training model
globalb3=1/numberofuser*sum(b3(:,:,:,bb(i,:)),4);
globalb4=1/numberofuser*sum(b4(:,:,bb(i,:)),3);

globalb5=1/numberofuser*sum(b5(:,:,bb(i,:)),3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   error2(i) = (error(i,1)/usernumber)*100;
  % Update the plot with the current accuracy
    plot(i, error2(i), 'bo-'); % 'bo-' plots a blue circle with a line connecting to the previous point
    % Use drawnow to update the figure window immediately
    drawnow;
end

save('error2_data.mat', 'error2');

% average y tipos de configuración

averageerror(average,kk)=error(iteration); % solo coge el último error
 %  end
averagedelay(average,kk)=sum(iterationtime);
averagedelayrandom(average,kk)=sum(iterationtimerandom);

display(i);
end
finalerror(kk,1)=sum(averageerror(:,kk))/averagenumber;
finaldelay(kk,1)=sum(averagedelay(:,kk))/averagenumber;
finaldelayrandom(kk,1)=sum(averagedelayrandom(:,kk))/averagenumber;
end


save CIFARresultbaseline3 error  finalerror finaldelay finaldelayrandom d

function [r,eta,sigma,T]=parameters(indices)
%%%%%%table 5.2.2.1-4 de 3GPP TS 38.214 version 16.5.0
efficacite=[0.0586 0.0977 0.1523 0.2344 0.3770 0.6016 0.8770 1.1758 1.4766 1.9141 2.4063 2.7305 3.3223 3.9023 4.5234];
F=512;%bit/paquet
RB=F./efficacite/26; 
% calculates the number of resource blocks (RB) required for transmitting 
% one packet of size F bits at each efficiency level.

for l=1:length(indices)
    r(l)=RB(indices(l));
end
eta=ones(1,length(r))*3;
sigma=ones(1,length(r));
end

=======
% function [encodedChromosome, decodedChromosome, scores] = test_adjustSizeDS(decodedChromosome, numberOfVariables, numberOfGenes, r, available_RBs, percentages)
%     % Normalización de los valores de 'r' y 'percentages'
%     normalizedR = r / sum(r);
%     normalizedPercentages = percentages / sum(percentages);
% 
%     % Double check the constraint after encoding
%     % while sum(decodedChromosome .* r) > available_RBs
%     while true
%         % Calculate relevance scores
%         effectivenessScores = normalizedPercentages ./ normalizedR;
%         scores=sum(effectivenessScores);
%         [~, order] = sort(effectivenessScores, 'ascend');  % Order to adjust by least cost effectiveness
% 
%         % Reduce the quantification level one step at a time, following the order of importance
%         for idx = order
%             currentStepSize = 1 / (2^5 - 1);  % smallest quantification unit
%             if decodedChromosome(idx) > currentStepSize
%                 decodedChromosome(idx) = decodedChromosome(idx) - currentStepSize;
%                 encodedChromosome = EncodeChromosome(decodedChromosome, numberOfVariables, numberOfGenes);
%                 if sum(decodedChromosome .* r) <= available_RBs
%                     break;  % If within limits, stop adjusting
%                 end
%             end
%         end
%     end
% end

function sortedTable = test_adjustSizeDS(inputTable, r)
    if width(inputTable) < 9
        error('Input table must have at least 9 columns.');
    end

    % Validate that the vector r has 8 elements
    if length(r) ~= 8
        error('Vector r must have 8 elements.');
    end

    % Extract numeric columns
    numericCols = varfun(@isnumeric, inputTable, 'OutputFormat', 'uniform');
    numericData = inputTable{:, numericCols};

    % Ensure there are at least 9 numeric columns
    if size(numericData, 2) < 9
        error('Table must have at least 9 numeric columns.');
    end

    % Create the new column by multiplying the first 8 numeric columns by the vector r
    newColumn = sum(numericData(:, 1:8) .* r, 2);

    % Append the new column to the numeric data
    numericDataWithNewColumn = [numericData, newColumn];

    % Sort the numeric data by the 9th column in descending order
    [~, sortOrder] = sort(numericDataWithNewColumn(:, 9), 'descend');

    % Reorganize the whole numeric data based on the sorted order
    sortedNumericData = numericDataWithNewColumn(sortOrder, :);

    % Recreate the table with the sorted numeric data
    tableSorted = inputTable;
    tableSorted{:, numericCols} = sortedNumericData(:, 1:end-1);

    % Add the new column to the sorted table
    newColumnName = 'NewColumn';
    tableSorted.(newColumnName) = sortedNumericData(:, end);

    % Reorder the rows of the original table based on the sort order
    sortedTable = tableSorted(sortOrder, :);
end
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
