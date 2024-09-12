function [imds1, imds2, imds3, imds4, imds5, imds6, imds7, imds8] = GetNonIIDCIFAR(rootFolder)
    categories = {'deer', 'dog', 'frog', 'cat', 'bird', 'automobile', 'horse', 'ship', 'truck', 'airplane'};
    imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
    
    % Initialize an array to store the output imageDatastore objects
    imdsArray = cell(1, 8);
    
    % Example: Non-IID proportions for each user
    proportions = [
        0.8 0.1 0.1 0   0   0   0   0   0   0;  % User 1: Mostly deer
        0   0.8 0.1 0.1 0   0   0   0   0   0;  % User 2: Mostly dog
        0   0   0.8 0.1 0.1 0   0   0   0   0;  % User 3: Mostly frog
        0   0   0   0.8 0.1 0.1 0   0   0   0;  % User 4: Mostly cat
        0   0   0   0   0.8 0.1 0.1 0   0   0;  % User 5: Mostly bird
        0   0   0   0   0   0.8 0.1 0.1 0   0;  % User 6: Mostly automobile
        0   0   0   0   0   0   0.8 0.1 0.1 0;  % User 7: Mostly horse
        0   0   0   0   0   0   0   0.8 0.1 0.1 % User 8: Mostly ship
    ];

    % Split the dataset for each user based on the proportions
    for user = 1:8
        imds_user = [];
        for category = 1:10
            % Filter images of the current category
            imds_cat = subset(imds, imds.Labels == categories{category});
            % Split the category-specific datastore
            numFiles = numel(imds_cat.Files);
            numUserFiles = round(proportions(user, category) * numFiles);
            imds_cat_user = splitEachLabel(imds_cat, numUserFiles / numFiles, 'randomized');
            imds_user = [imds_user; imds_cat_user.Files];
        end
        % Create imageDatastore for user
        imdsArray{user} = imageDatastore(imds_user, 'LabelSource', 'foldernames');
    end
    
    % Assign the output variables
    [imds1, imds2, imds3, imds4, imds5, imds6, imds7, imds8] = imdsArray{:};
end
