%% COMPARISON WITH BEST ESTIMATED PI

iteration=300;
averagenumber=1;
verboseFL = false;
miniBatchSize = 100;
executionEnviroment ='parallel';
AccDevMat=true;
Shuffle='never';
long=true;

% 1 (1), 3/4 (2), 1/2 (3),  1/4 (4), 1/8 (5), 1/16 (6)s

%% RUCHE OR NOT
ruche = true;
if ruche
      if ~long
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      else
          refModelName = 'Ref_Model_70_i_3_r_noQ';     
          directory_RefModel = 'refModel_70_i_3_avg'; 
      end  
      baseDir = '/gpfs/workdir/costafrelu/Sketch.m/';  
else
      if ~long
          refModelName = 'Ref_Model_15_i_15_r_noQ';
          directory_RefModel = 'refModel_15_i_15_withFragSDS';  
      else
          refModelName = 'Ref_Model_70_i_3_r_noQ';     
          directory_RefModel = 'refModel_70_i_3_avg'; 
      end  
      baseDir = fullfile('..', 'OptimizationOfPI', 'Opt_noOpt','WithAndWithoutDSFrag');
end

directory_tempDir = sprintf('RefMod_temporaryDir_i_%d_avg_%d', iteration, averagenumber); % para ACC y DEV este es el nombre de la ULTIMA CARPETA
directory_baseDir = sprintf('RefMod_i_%d_avg_%d', iteration, averagenumber);

fullpath_tempDir = fullfile('/gpfs/workdir/costafrelu/temporaryMat/', directory_tempDir);
if ~exist(fullpath_tempDir, 'dir')
    mkdir(fullpath_tempDir);
end
fullpath_baseDir2=fullfile(baseDir,directory_baseDir);
if ~exist(fullpath_baseDir2, 'dir')
    mkdir(fullpath_baseDir2);
end

FragSDS = 1;
% percentages =[1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8];
% percentages =[1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2];
% percentages = [
%     1,1,1,1,1,1,1,1;
%     1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2;
%     ];
percentages = [1,1,1,1,1,1,1,1];

% chromosomes = [
%     1, 1, 0.6774, 0.2581, 0.9355, 0.9677, 0.7097, 0.2258;
%     1, 1, 0.5806, 0.4194, 1, 1, 0.8387, 0.6450;
%     1, 0.3226, 0.1290, 0.8380, 0.8065, 0.9677, 0.5806, 0.1935;
%     1, 0.8710, 0.1290, 0.2581, 1, 0.8087, 0.4516, 0.4839;
%     1, 0.7420, 0.4194, 0.3230, 1, 0.8390, 0.4190, 0.1610;
%     1, 0.9680, 0.4200, 0.3230, 0.9350, 0.3230, 0.1930, 0.4840;
%     1, 0.8060, 0.2580, 0.2580, 0.4820, 0.3230, 0.4516, 0.4840;
%     0.9677, 0.4839, 0.6452, 0, 1, 1, 0.1290, 0.2258;
%     0.2258, 0.9680, 0.4840, 0, 0.9030, 1, 0.4516, 0.4840;
%     1, 0.8387, 0, 0.1300, 1, 0.5160, 0.0645, 0.4194
% ];

% chromosomes = [
%     0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9;
%     0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7;
%     0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,;
% ];

chromosomes = [1,1,1,1,1,1,1,1];


[accuracyFitness]  = runFLEnviroment_RUCHE_sameDS_REF(iteration, averagenumber,... %REFERENCE
    verboseFL, miniBatchSize, executionEnviroment, AccDevMat, Shuffle, directory_RefModel, FragSDS, percentages, fullpath_tempDir, fullpath_baseDir2); 


