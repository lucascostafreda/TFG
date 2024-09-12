%% COMPARISON WITH BEST ESTIMATED PI

  CQI_indices = [12, 10, 8, 6, 12, 10, 8, 6];
  [r, eta, sigma] = parameters(CQI_indices);
  RB_usedMatrix = [];

<<<<<<< HEAD
iteration = 1;
averagenumber = 1;
verboseFL = true;
=======
iteration = 5;
averagenumber = 5;
verboseFL = false;
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
miniBatchSize = 100;
executionEnviroment ='parallel';
AccDevMat=true;
Shuffle='every-epoch';
long=false;

% 1 (1), 3/4 (2), 1/2 (3),  1/4 (4), 1/8 (5), 1/16 (6)s

%% RUCHE OR NOT
ruche = true;
if ruche
      if ~long
<<<<<<< HEAD
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
=======
          % refModelName = 'Ref_Model_15_i_15_r_noQ';
          % directory_RefModel = 'refModel_15_i_15_withFragSDS';  
          refModelName = 'Ref_Model_5_i_5_avg';
          directory_RefModel = 'RefMod_i_5_avg_5_noniid';  
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
      else
          refModelName = 'Ref_Model_70_i_3_r_noQ';     
          directory_RefModel = 'refModel_70_i_3_avg'; 
      end  
      % baseDir = '/gpfs/workdir/costafrelu/Sketch.m/';  
else
      if ~long
<<<<<<< HEAD
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      else
          refModelName = 'Ref_Model_70_i_3_r_noQ';     
          directory_RefModel = 'refModel_70_i_3_avg'; 
=======
          % refModelName = 'Ref_Model_5_i_5_r_noQ';
          refModelName = 'Ref_Model_5_i_5_r_noQ';
          % directory_RefModel = 'refModel_5_i_5_av';  
          directory_RefModel = 'RefMod_i_5_avg_5';  
      else
          refModelName = 'Ref_Model_300_i_2_r_noQ';     
          directory_RefModel = 'RefMod_temporaryDir_i_300_avg_2'; 
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
      end  
      % baseDir = fullfile('..', 'OptimizationOfPI', 'Opt_noOpt','WithAndWithoutDSFrag');
end

<<<<<<< HEAD
directory_tempDir = sprintf('0.9PI_100%_accuracyConvergens_i_%d_avg_%d', iteration, averagenumber); % para ACC y DEV este es el nombre de la ULTIMA CARPETA
FragSDS = 1;

percentages = [1,1,1,1,1,1,1,1];

chromosomes = [
    1, 1, 1, 1, 1, 1, 1, 1;
    0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9;
    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8;
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6; 
];

for k = 1:size(percentages,1) 
    directory_tempDir = sprintf('%s_percentage_%d', directory_tempDir, k);
=======
directory_tempDir = sprintf('SameFullPI_DiffUnifDS_noniidRefMod_i_%d_avg_%d', iteration, averagenumber); % para ACC y DEV este es el nombre de la ULTIMA CARPETA
FragSDS = 1;

percentages = [
    % 3/4, 3/4, 3/4, 3/4, 3/4, 3/4, 3/4, 3/4;
    1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2;
    1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4;
    1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8;
    % 1,1,1,1,1,1,1,1;
    % 1/8, 1/8, 1,1, 1/8, 1/8, 1, 1;
    % 1/2, 3/4, 1,1, 1/2, 3/4, 1, 1;
    % 1/8, 1/2, 3/4, 1, 1/8, 1/2, 3/4, 1;
    ];

chromosomes = [
    % 0,0,0,1,0,0,0,0;
    % 1, 1, 1, 0, 1, 1, 1, 0; %Sacrifico el dispositivo más potente
    % 0,0,0.75,1,0,0,0.75,1; %Sacrifico con mejores enlaces (caben menos dispositivos) 
    % 0.9, 0.75, 0.6, 0.4, 0.9, 0.75, 0.6, 0.4; %LEGAL, bueno 
    % 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55; % todos por igual y cerca del límite
    % 0.3, 0.3, 0.5, 0.7, 0.3, 0.3, 0.5, 0.7; % Quiero incluir a más dispositivos a coste de bajar las tasas
    % 0, 0.4, 0.6, 0.7, 0, 0.4, 0.6, 0.7; %Sacrificando uno
    1, 1, 1, 1, 1, 1, 1, 1; %MEJOR PERO ILEGAL
    ];
    % 1, 1, 1, 1, 1, 1, 1, 1
    % 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9;
    % 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8;
    % 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6;
    % ];

for k = 1:size(percentages,1) 
    directory_tempDir = sprintf('%s_percentage_%d', directory_tempDir, k);
    curretPercentages = percentages(k, :);
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    for j = 1:size(chromosomes, 1)  % Iterate over each chromosome configuration
        currentChromosome = chromosomes(j, :);
        % filename = sprintf('fitness_%d_i_%d_av_chromosome%d.mat', iteration, averagenumber, j);
        % fullFilename = fullfile(baseDir, filename);
        [accuracyFitness, RB_used] = runFLEnviroment_RUCHE_sameDS_withDev(currentChromosome, iteration, averagenumber, r,...
<<<<<<< HEAD
            ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, refModelName, directory_RefModel, directory_tempDir, FragSDS, percentages, j, k);
=======
            ruche, verboseFL, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, refModelName, directory_RefModel, directory_tempDir, FragSDS, curretPercentages, j, k);
>>>>>>> 9c6ccc124 (Reinicializando el repositorio)
    end
end
  