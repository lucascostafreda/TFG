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
[imds1, imds2, imds3, imds4] = splitEachLabel(imds, 0.25, 0.25, 0.25, 0.25);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

averagenumber=1;  % Average number of runing simulations.  Repeticiones, para promediar el error del pack al final !!
iteration=250;     % Total number of global FL iterations.
usernumber=4;
%proposed=1;
modelEpoch = 3;
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

P=1;
I=([0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15]-0.04)*0.001;  
alpha=0.85;
RBnumber=length(I); %OJO, puede cambiar
learningrate=0.008;
varSize = 32; %tamaño imagen

for userno=4:2:4    % Number of users.
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

SINR=P*1*(d(1:usernumber,1).^(-2))./I;
rate=2*log2(1+SINR); 
time=116906*32./rate/1024/1024; 

accuracy=zeros(iteration,1); 
gradient=zeros(usernumber,1);
wupdate=zeros(iteration,usernumber);   % local model for each user
totalGradient = zeros(iteration, 1);

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
    'Shuffle', 'every-epoch',... % mezclar antes de cada epoch, no sirve de mucho, pues pasasuna vez
    'Verbose', true);%,... 
    %'Plots', 'training-progress'); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize the figure for plotting
fig1=figure;
hold on; 
 
xlabel('Iteration');
ylabel('Current Accuracy');
title('Real-Time Accuracy Plot');
grid on;

fig2=figure;
hold on

xlabel('Iteration');
ylabel('Current Gradient');
title('Real-Time Gradient Plot');
grid on;

kkkk=0;

for i=1:1:iteration
    
%%%%%%%%%%%%%%%%%%%%%%%Setting of local FL model %%%%%%%%%%%%%%%%%%%%%%%%%%   
error2 = zeros(iteration, 1); % Pre-allocate para eficiencia

for user=1:1:usernumber  
    
    clear netvaluable;
    Winstr1=strcat('net',int2str(user));     
    midstr=strcat('imds',int2str(user)); 
    %creates a string that refers to the ImageDatastore specific to the current user.
     
    eval(['imdss','=',midstr,';']);
    % string as MATLAB code, assigning the user-specific ImageDatastore to the variable imdss
    % allows selecting the correct dataset for each user during the iteration.

% if i>1 
if mod(i, modelEpoch) == 1 && i~=1   
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


%%%%%%%%%%%%%%%%%%%calculate identification accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels = classify(netvaluable, imds_test);

confMat = confusionmat(imds_test.Labels, labels);
%figure;
% confusionchart(confMat);
% title('Gráfico de Matriz de Confusión No Normalizada');
%disp(confMat);
confMat = confMat./sum(confMat,2);
% figure;
% confusionchart(confMat, 'Normalization', 'row-normalized');
% title('Gráfico de Matriz de Confusión Normalizada');

% disp(confMat);
% disp(diag(confMat));
% error3 = mean(diag(confMat));
% fprintf('Error medio en la iteración %d usuario %d: %.4f\n', user, i, error3); %display of mean(diag(confMat))
accuracy(i,1)=mean(diag(confMat))+accuracy(i,1); 
%fprintf('Error en la iteración %d: %.4f\n', i, error(i,1));


accuracy2(i) = (accuracy(i,1)/usernumber)*100;

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

% if i==1    
% 
%     deviationw1all(:,:,:,:,user)= w1(:,:,:,:,user);
%     deviationw2all(:,:,:,:,user)=w2(:,:,:,:,user);
%     deviationw3all(:,:,:,:,user)= w3(:,:,:,:,user);
%     deviationw4all(:,:,user)=w4(:,:,user);
%     deviationw5all(:,:,user)=w5(:,:,user);
% 
%     deviationb1all(:,:,:,user)=b1(:,:,:,user);
%     deviationb2all(:,:,:,user)=b2(:,:,:,user);
%     deviationb3all(:,:,:,user)=b3(:,:,:,user);
%     deviationb4all(:,:,user)=b4(:,:,user);
%     deviationb5all(:,:,user)= b5(:,:,user);
% 
% else
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

%%%%%%%%%%%%%%%%%%%%%%% user selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

        for jjj=1:1:usernumber % all deviations x user
            gradient(jjj,1)=sum(sqrt(deviationw1all(:,:,:,:,jjj).^2),'all')+sum(sqrt(deviationw2all(:,:,:,:,jjj).^2),'all')+sum(sqrt(deviationw3all(:,:,:,:,jjj).^2),'all')...
                            +sum(sqrt(deviationw4all(:,:,jjj).^2),'all')+sum(sqrt(deviationw5all(:,:,jjj).^2),'all')+sum(sqrt(deviationb1all(:,:,:,jjj).^2),'all')...
                            +sum(sqrt(deviationb2all(:,:,:,jjj).^2),'all')+sum(sqrt(deviationb3all(:,:,:,jjj).^2),'all')...
                            +sum(sqrt(deviationb4all(:,:,jjj).^2),'all')+sum(sqrt(deviationb5all(:,:,jjj).^2),'all');
        end

        totalGradient(i)=sum(gradient);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ENVÍO PARAM.
% Si no, pasa directamente a la siguiente iteración
    
    if mod(i,modelEpoch) == 0
    
    % Empieza todo el proceso para actualizar el modelo global
    % 1. Procesar las desviaciones
    % 2. Codificarlas y decodificarlas
    % 3. Usar estas para actualizar el modelo global

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
        
            % if i==1 %% ojo, estto no lo vamos a necesitar, nunca será i==1
            %     w1(:,:,:,:,user) = deviationw1;
            %     w2(:,:,:,:,user) = deviationw2;
            %     w3(:,:,:,:,user) = deviationw3;
            %     w4(:,:,user) = deviationw4;
            %     w5(:,:,user) = deviationw5;
            % 
            %     b1(:,:,:,user) = deviationb1;
            %     b2(:,:,:,user) = deviationb2;
            %     b3(:,:,:,user) = deviationb3;
            %     b4(:,:,user) = deviationb4;
            %     b5(:,:,user) = deviationb5;
            % 
                      
            %else       % global, la primera vez que se entra al bucle es = 0
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
           
            %end
        
    end
    % Solo 1 vez por iteracion
     %%%%%%%%%%%%%%%% update the global FL model  %%%%%%%%%%%%  
    
        globalw1 = 1/numberofuser * sum(w1, 5);  % global training model
        globalw2 = 1/numberofuser * sum(w2, 5);  % global training model
        globalw3 = 1/numberofuser * sum(w3, 5);
        globalw4 = 1/numberofuser * sum(w4, 3);  
        globalw5 = 1/numberofuser * sum(w5, 3);
    
        globalb1 = 1/numberofuser * sum(b1, 4);  
        globalb2 = 1/numberofuser * sum(b2, 4); 
        globalb3 = 1/numberofuser * sum(b3, 4);
        globalb4 = 1/numberofuser * sum(b4, 3);
        globalb5 = 1/numberofuser * sum(b5, 3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(fig1);
plot(i, accuracy2(i), 'bo-'); 

figure(fig2);
plot(i, totalGradient(i), 'ro-');

drawnow;

end
save('gradient_vector.mat', 'totalGradient');
save('accuracy2_data.mat', 'accuracy2');

% average y tipos de configuración
% averageerror(average,kk)=error(iteration); % solo coge el último error
% averagedelay(average,kk)=sum(iterationtime);
% averagedelayrandom(average,kk)=sum(iterationtimerandom);

end
% finalerror(kk,1)=sum(averageerror(:,kk))/averagenumber;
% finaldelay(kk,1)=sum(averagedelay(:,kk))/averagenumber;
% finaldelayrandom(kk,1)=sum(averagedelayrandom(:,kk))/averagenumber;
end



