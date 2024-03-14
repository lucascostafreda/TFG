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
[imds1, imds2, imds3, imds4, imds5, imds6, imds7, imds8] = splitEachLabel(imds, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

averagenumber=1;  % Average number of runing simulations.  Repeticiones, para promediar el error del pack al final !!
iteration=30;     % Total number of global FL iterations.
usernumber = 8;
learningrate=0.008;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
total_available_RBs = 80;

% Static definition of CQI
%CQI_indices = randi([1 15], 1, usernumber);
CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
[r, eta, sigma] = parameters(CQI_indices);
PI = [1,0,0,0,0,0,0,0];
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

accuracy=zeros(iteration,1); 
gradient=zeros(usernumber,1);
wupdate=zeros(iteration,usernumber);   % local model for each user
totalGradient = zeros(iteration, 1);
%energyCost = zeros(iteration, 1);
energyCost=0;
RB_used = zeros(iteration,1);
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
title('Real-Time Accuracy Plot (CM_NE)');
grid on;

fig2=figure;
hold on

xlabel('Iteration');
ylabel('Current Gradient');
title('Real-Time Gradient Plot (CM_NE)');
grid on;

fig3=figure;
hold on; 
 
xlabel('Iteration');
ylabel('Energy cost');
title('Real-Time Energy Cost Plot (CM_NE)');
grid on;

kkkk=0;
dataStorage = struct([]); % Inicializa un array de estructuras vacío

for i=1:1:iteration

% CQI_indices = randi([1 15], 1, usernumber);
% [r, eta, sigma] = parameters(CQI_indices);
% inverse_r = 1 ./ r;
% PI = inverse_r / sum(inverse_r);

% % Almacena las variables en el array de estructuras
%     dataStorage(i).CQI_indices = CQI_indices;
%     dataStorage(i).r = r;
%     dataStorage(i).PI = PI;

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
accuracy(i,1)=mean(diag(confMat))+accuracy(i,1); 
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

%%%%%%%%%%%%%%%%%%%%% totalGrad x User calculation %%%%%%%%%%%%%%%%%%%%%%%%%%

        for jjj=1:1:usernumber % all deviations x user
            gradient(jjj,1)=sum(sqrt(deviationw1all(:,:,:,:,jjj).^2),'all')+sum(sqrt(deviationw2all(:,:,:,:,jjj).^2),'all')+sum(sqrt(deviationw3all(:,:,:,:,jjj).^2),'all')...
                            +sum(sqrt(deviationw4all(:,:,jjj).^2),'all')+sum(sqrt(deviationw5all(:,:,jjj).^2),'all')+sum(sqrt(deviationb1all(:,:,:,jjj).^2),'all')...
                            +sum(sqrt(deviationb2all(:,:,:,jjj).^2),'all')+sum(sqrt(deviationb3all(:,:,:,jjj).^2),'all')...
                            +sum(sqrt(deviationb4all(:,:,jjj).^2),'all')+sum(sqrt(deviationb5all(:,:,jjj).^2),'all');
        end

        totalGradient(i)=sum(gradient);

%%%%%%%%%%%%%%%%%%%%%%%%% User selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% selecciono según distribución de los recursos
% OJO, PI aquí está totalmente comprometida a r. 
% Ahora será PI la que cambie iterativamente

        available_RBs = total_available_RBs;
        [selected_users, total_resource_usage] = selectUsersBasedOnPI(PI, r, usernumber);
        RB_used(i) = total_resource_usage;
        
        %Vect_producto = PI(selected_users).*r(selected_users);
        %energyCost(i) = sum(Vect_producto);
        
        energyCost = sum(PI(selected_users).*r(selected_users));
        
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
        

%%%%%%%%%%%%%%%%%%%%%%%%% PRINT RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

         %fprintf('CQI_indices: %s\n', mat2str(CQI_indices));
        % fprintf('r: %s\n', mat2str(r, 4)); % 4 dígitos de precisión para 'r'
         %fprintf('PI: %s\n', mat2str(PI, 4)); % 4 dígitos de precisión para 'PI'
        % fprintf('Vect_producto: %s\n', i, mat2str(Vect_producto, 4));
         fprintf('Energy Cost: %0.2f\n', energyCost); % Mostrar el Energy Cost

figure(fig1);
plot(i, accuracy2(i), 'bo-'); 

figure(fig2);
 if i > 9 % Skip the first nine iterations
    plot(i, totalGradient(i), 'ro-');
 end 

figure(fig3);
plot(i, RB_used(i), 'bo-');

drawnow;

end
% serviceCost = abs(totalGradient-optimalGradient) ;
% disp('The service cost of the policy PI is:');
% disp(serviceCost);
average_RB=mean(RB_used);
disp(average_RB)
% save('gradient_vector.mat', 'totalGradient');
% save('accuracy2_data.mat', 'accuracy2');
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [selected_users, total_resource_usage] = selectUsersBasedOnPI(PI, r, num_users)
    %The restriction is already met sum(PI.*r)<=R
    selected_users = [];
    total_resource_usage = 0;
    % Independently select users based on PI, ensuring resource constraints are respected
    for user = 1:num_users
        if rand() <= PI(user)
            selected_users = [selected_users, user];
            total_resource_usage = total_resource_usage + r(user);
        end
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OJO, selects users based on their resource efficiency or another metric 
% until adding another user would exceed the limit.

% function selected_users = selectUsersGreedy(users, available_RBs, r)
%     % Sort users by their resource requirements (ascending order for this example)
%     [sorted_r, sorted_indices] = sort(r, 'ascend');
% 
%     selected_users = [];
%     current_RBs = 0;
% 
%     for i = 1:length(users)
%         user_idx = sorted_indices(i);
%         if (current_RBs + sorted_r(i)) <= available_RBs
%             % Add user if total RBs won't exceed the limit
%             selected_users = [selected_users, user_idx];
%             current_RBs = current_RBs + sorted_r(i);
%         else
%             % If adding the next user exceeds the limit, break the loop
%             break;
%         end
%     end
% end

% function selected_users = optimizeSelection(selected_users, r, available_RBs)
%     current_usage = sum(r(selected_users));
%     remaining_RBs = available_RBs - current_usage;
%     all_users = 1:length(r);
%     unselected_users = setdiff(all_users, selected_users);
% 
%     % Sort unselected users by how much they 'fill up' the remaining RBs, descending
%     [sorted_diffs, sort_idx] = sort(r(unselected_users) - remaining_RBs, 'descend');
%     sorted_unselected_users = unselected_users(sort_idx);
% 
%     for i = 1:length(sorted_unselected_users)
%         user_to_consider = sorted_unselected_users(i);
%         if r(user_to_consider) <= remaining_RBs
%             % If adding this user does not exceed the total RBs, add them
%             selected_users = [selected_users, user_to_consider];
%             remaining_RBs = remaining_RBs - r(user_to_consider);
%             if remaining_RBs == 0
%                 % Perfectly utilized all RBs
%                 break;
%             end
%         end
%     end
% 
%     % Attempt to swap if possible (very basic swap logic for illustration)
%     if remaining_RBs > 0
%         for i = 1:length(selected_users)
%             for j = 1:length(sorted_unselected_users)
%                 if r(sorted_unselected_users(j)) - r(selected_users(i)) <= remaining_RBs
%                     % Perform a swap if it increases utilization without exceeding limit
%                     temp_selection = selected_users;
%                     temp_selection(i) = sorted_unselected_users(j); % Swap
%                     new_usage = sum(r(temp_selection));
%                     if new_usage <= available_RBs && new_usage > current_usage
%                         selected_users = temp_selection; % Accept swap
%                         current_usage = new_usage;
%                         remaining_RBs = available_RBs - new_usage;
%                         break; % Break inner loop
%                     end
%                 end
%             end
%         end
%     end
% end


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

function [r,eta,sigma]=parameters(indices)
    %%%%%%table 5.2.2.1-4 de 3GPP TS 38.214 version 16.5.0
    efficacite=[0.0586 0.0977 0.1523 0.2344 0.3770 0.6016 0.8770 1.1758 1.4766 1.9141 2.4063 2.7305 3.3223 3.9023 4.5234];
    F=512;%bit/paquet
    RB=F./efficacite/26; 

    for l=1:length(indices)
        r(l)=RB(indices(l));
    end

    eta=ones(1,length(r))*3;
    sigma=ones(1,length(r));
end


