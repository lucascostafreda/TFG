%% COMPARISON WITH BEST ESTIMATED PI

  CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
  [r, eta, sigma] = parameters(CQI_indices);
  RB_usedMatrix = [];

iteration = 200;
averagenumber = 1;
verboseFL = true;
miniBatchSize = 100;
executionEnviroment ='parallel';
AccDevMat=true;
Shuffle='every-epoch'; %OJOOOO
long=true;

% 1 (1), 3/4 (2), 1/2 (3),  1/4 (4), 1/8 (5), 1/16 (6)s

%% RUCHE OR NOT
ruche = true;
if ruche
      if ~long
          refModelName = 'Ref_Model_5_i_5_avg';
          directory_RefModel = 'RefMod_i_5_avg_5_noniid';
      else
          % refModelName = 'Ref_Model_70_i_3_r_noQ';     
          % directory_RefModel = 'refModel_70_i_3_avg'; 

          refModelName = 'Ref_Model_250_i_3_r_noQ';     
          directory_RefModel = 'RefMod_i_250_avg_3_noniid'; 

      end  
      % baseDir = '/gpfs/workdir/costafrelu/Sketch.m/';  
else
      if ~long
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      else
          % refModelName = 'Ref_Model_70_i_3_r_noQ';     
          % directory_RefModel = 'refModel_70_i_3_avg'; 
          refModelName = 'Ref_Model_250_i_3_r_noQ';     
          directory_RefModel = 'RefMod_i_250_avg_3_noniid';
      end  
      % baseDir = fullfile('..', 'OptimizationOfPI', 'Opt_noOpt','WithAndWithoutDSFrag');
end

directory_tempDir = sprintf('PISample_0.6_1_i_%d_avg_%d', iteration, averagenumber); % para ACC y DEV este es el nombre de la ULTIMA CARPETA
% directory_tempDir = sprintf('DiffIrregPI_SameIrregDS_i_%d_avg_%d_2_v2', iteration, averagenumber)
% directory_tempDir = sprintf('SameFullPI_DiffUniDS_i_%d_avg_%d_2_v2', iteration, averagenumber);
FragSDS = 1;
    
percentages = [
    % 1, 1, 1, 1, 1, 1, 1, 1;
    1/8, 1/2, 3/4, 1, 1/8, 1/2, 3/4, 1;
    % 3/4, 3/4, 3/4, 3/4, 3/4, 3/4, 3/4, 3/4;
    % 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2;
    % 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8;
    ];

chromosomes = [
    % 1, 1, 1, 0, 1, 1, 1, 0; %Sacrifico el dispositivo más potente
     % 0,0,0.75,1,0,0,0.75,1; %Sacrifico con mejores enlaces (caben menos dispositivos) 
    % 0.9, 0.75, 0.6, 0.4, 0.9, 0.75, 0.6, 0.4; %LEGAL, bueno 
    % 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55; % todos por igual y cerca del límite
    % 0.3, 0.3, 0.5, 0.7, 0.3, 0.3, 0.5, 0.7; % Quiero incluir a más dispositivos a coste de bajar las tasas
     % 1, 1, 1, 1, 1, 1, 1, 1;
    % 0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8;

    % Post evaluation
    0.7419, 0.8387, 0.4516, 0.4839, 0.7419, 0.8387, 0.4516, 0.4839; % --> Fitness 0.6454, RB=448.58
    % 0.7419, 0.4839, 0.5484, 0.5484, 0.7419, 0.4839, 0.5484, 0.5484  % --> Fitness 0.6533, RB=449.56
    % 
    % 0.1935, 0.8387, 0.9355, 0.3548, 0.1935, 0.8387, 0.9355, 0.3548; % --> Fitness 0.7136, RB 447.68
    % 0.8065, 0.9355, 0.6452, 0.3226, 0.8065, 0.9355, 0.6452, 0.3226; % --> Fitness 0.73, RB 441.65

    % 0.9677, 0.8710, 0.7742, 0.2258, 0.9677, 0.8710, 0.7742, 0.2258; % --> Fitness 0.8367, RB 435.56
    % 0.7419, 0.7742, 0.6774, 0.3871, 0.7419, 0.7742, 0.6774, 0.3871; % --> Fitness 0.83, RB 446,98
    % 1, 0.87, 0.7419, 0.258, 1, 0.87, 0.7419, 0.258 % --> Fitness 0.8289, RB 444,54
    % 
];

for k = 1:size(percentages,1) 
    directory_tempDir = sprintf('%s_percentage_%d', directory_tempDir, k);
    for j = 1:size(chromosomes, 1)  % Iterate over each chromosome configuration
        currentChromosome = chromosomes(j, :);

        [accuracyFitness, RB_used] = runFLEnviroment_RUCHE_sameDS_withDev(currentChromosome, iteration, averagenumber, r,...
            ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, refModelName, directory_RefModel, directory_tempDir, FragSDS, percentages, j, k);
    end
end
  