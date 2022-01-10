%starter code for project 4: Convolutional Neural Network
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2017
%Bharadwaj Ravichandran, Jan 2020
%Shimian Zhang, Feb 2021

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Purushartha Singh
    PSU Email ID: pxs288@psu.edu 
    Description: THis file contains all the parts of the project 4. The
    first section is the input storage followed by the default network
    given in the starter code. This is followed by a skinny and wide
    network and finally the Alex Net transfoer learning input and network
    code.
    The Last two sections have visualization and accuracy functions which
    can be used for all the networks.
%}

dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

train_folder = 'train';
test_folder  = 'test';
% uncomment after you create the augmentation dataset
% train_folder = 'train_aug';
% test_folder  = 'test_aug';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%% Global Variables
rng('default');
numEpochs = 5; % 5 for both learning rates
batchSize = 25;
nTraining = length(train.Labels);
%% (Part 2) Keep commented unless you want to remake augmented data
% augment(Symmetry_Groups, 0);
augment(Symmetry_Groups, 1);


%% Define the Network Structure, Default Network
% lines such as the example at the bottom of the code
%  CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 
layers = [
    imageInputLayer([128 128 1]); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(5,20,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    fullyConnectedLayer(25); % Fullly connected layer with 50 activations
    dropoutLayer(.25); % Dropout layer
    fullyConnectedLayer(17); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];
%% Section for setting training options and running the CNN (Def Network)
if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% Set the training options
options = trainingOptions('sgdm','MaxEpochs',15,... 
    'InitialLearnRate',5e-5,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'OutputFcn',@plotTrainingAccuracy,... 
    'MaxEpochs',numEpochs,...
    'ExecutionEnvironment', 'gpu');
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net1,info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

%% Part 3 skinny
% 
% lines such as the example at the bottom of the code
%  (CONV -> Norm -> ReLU -> POOL )*4-> FC -> DROPOUT -> FC -> SOFTMAX 
layers = [
    imageInputLayer([128 128 1]); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(7, 80, 'Padding', [2 2], 'Stride', [1 1]);  % 80 7 x 7 Filters. 
    batchNormalizationLayer;
    reluLayer();  % ReLU layer 
    %maxPooling2dLayer(2, 'Stride', 2); % 2 x 2 maxpool
    
    convolution2dLayer(7,60 ,'Padding', [2 2],'Stride', [2 2]); % 60 7 x 7 Filters. 
    batchNormalizationLayer;
    reluLayer();
    %maxPooling2dLayer(2, 'Stride', 2); % 2 x 2 maxpool. 
    
    convolution2dLayer(5,40 ,'Padding', [2 2],'Stride', [2 2]); % 40 7 x 7 Filters.
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2, 'Stride', 2); % 2 x 2 maxpool.
    
    
    convolution2dLayer(5,32 ,'Padding', [2 2],'Stride', [2 2]); % 20 7 x 7 Filters. 
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2, 'Stride', 2); % 2 x 2 maxpool. 
    
    fullyConnectedLayer(300); % FC layer with 300 units
    dropoutLayer(.25); % Dropout layer
    
    fullyConnectedLayer(17); % FC layer with 17 units.
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];

if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% Set the training options
options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',1e-5,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'OutputFcn',@plotTrainingAccuracy,... 
    'MaxEpochs',10,...
    'ExecutionEnvironment', 'gpu');
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
t = tic;
[net1,info1] = trainNetwork(train,layers,options);
% load('./modelCheckpoints/net_checkpoint__6120__2021_04_12__18_23_33.mat', 'net1');

options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',5e-5,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'OutputFcn',@plotTrainingAccuracy,... 
    'MaxEpochs',10,...
    'ExecutionEnvironment', 'gpu');
[net1,info1] = trainNetwork(train,net1.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

% Test on the validation data
YTest = classify(net1,val);
val_acc = mean(YTest==val.Labels)

figure1 = figure;
plotTrainingAccuracy_All(info1,numEpochs);
%% Part 3 wide
layers = [
    imageInputLayer([128 128 1]); % Input to the network is a 128x128x1 sized image
    convolution2dLayer(9, 100, 'Padding', [2 2], 'Stride', [2 2]);  % 100 9 x 9 Filters.  
    batchNormalizationLayer;
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2, 'Stride', 2); % 2 x 2 maxpool. 
    
    convolution2dLayer(7,80 ,'Padding', [2 2],'Stride', [2 2]); % 80 7 x 7 Filters.
    batchNormalizationLayer,
    reluLayer();
    maxPooling2dLayer(2, 'Stride', 2); % 2 x 2 maxpool.
    
    fullyConnectedLayer(300); % FC layer with 300 units
    dropoutLayer(.25); % Dropout layer
    
    fullyConnectedLayer(17); % FC layer with 17 units.
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];

if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% Set the training options
options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',5e-6,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'OutputFcn',@plotTrainingAccuracy,... 
    'MaxEpochs',10,...
    'ExecutionEnvironment', 'gpu');
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net1,info1] = trainNetwork(train,layers,options);
options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',5e-5,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'OutputFcn',@plotTrainingAccuracy,... 
    'MaxEpochs',10,...
    'ExecutionEnvironment', 'gpu');
[net1,info1] = trainNetwork(train,net1.Layers,options);

fprintf('Trained in in %.02f seconds\n', toc(t));

% Test on the validation data
YTest = classify(net1,val);
val_acc = mean(YTest==val.Labels)

figure1 = figure;
plotTrainingAccuracy_All(info1,numEpochs);
%% Transfer learning through Alexnet - Changing the input
alexAugment('./data/wallpapers/train_aug/', './data/wallpapers/train_alex/', Symmetry_Groups)
alexAugment('./data/wallpapers/test_aug/', './data/wallpapers/test_alex/', Symmetry_Groups)

%% Alexnet (Part 4) Load data
train_folder = 'train_alex';
test_folder  = 'test_alex';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%% Alexnet CNN
net = alexnet;

layersTransfer = net.Layers(1:end-3);


layers = [
    layersTransfer
    fullyConnectedLayer(17,'WeightLearnRateFactor',40,'BiasLearnRateFactor',40); % FC layer with 17 units.
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];
options = trainingOptions('adam','MaxEpochs',15,... 
    'InitialLearnRate',5e-5,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', 50, ...
    'MaxEpochs',10, ...    
    'ExecutionEnvironment', 'gpu', ...
    'OutputFcn',@plotTrainingAccuracy);

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net1, info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

%% Test on the test data and visualizations (Part 5)

info2.TrainingLoss = [5.6151, 4.9381, 3.4705, 4.6439, 3.8727, 3.8636, 3.4506, 3.0619, 3.1918, 3.3982, 3.2585, 3.5768, 2.9253, 3.1806, 2.956, 3.2535, 2.8742, 2.3561, 3.1311, 2.6556, 2.8825, 3.2407, 2.6871, 3.1013, 2.7554, 2.3974, 3.2891, 2.7314, 2.0724, 2.7109, 2.1209, 2.3181, 2.3697, 2.1276, 2.6231, 2.1526, 2.5287, 3.0724, 2.3217, 2.1466, 2.3388, 2.5142, 2.1819, 2.1757, 2.2491, 2.6739, 2.6564, 1.8529, 2.014, 2.1852, 2.1939, 2.0814, 2.2227, 1.8856, 1.7519, 2.4411, 2.0012, 1.8207, 2.3115, 1.9226, 1.9214, 1.9864, 1.9601, 2.1042, 2.0262, 1.6019, 1.5766, 1.8321, 1.9929, 2.0622, 1.3768, 1.3766, 1.4535, 1.3989, 1.478, 1.3089, 1.5397, 1.395, 1.0799, 1.1997, 1.1817, 1.2669, 1.2376, 1.0809, 0.9275, 1.2133, 0.6892, 1.1214, 0.7511, 0.6103, 0.9299, 1.0121, 1.023, 0.5899, 0.7809, 0.7648, 0.7091, 0.8431, 0.8721, 0.6989, 0.5315, 0.7478, 0.5525, 0.5257, 0.4902, 0.35, 0.4391, 0.6475, 0.5127, 0.5208, 0.4422, 0.3124, 0.5549, 0.5261, 0.3648, 0.5819, 0.5354, 0.2151, 0.4687, 0.7043, 0.4667, 0.2918, 0.471, 0.3017, 0.3134, 0.4598, 0.2232, 0.4244, 0.176, 0.3284, 0.2233, 0.4133, 0.4717, 0.1106, 0.3124, 0.2295, 0.3025, 0.2392, 0.2678, 0.1914, 0.1426, 0.3616, 0.2498, 0.1527, 0.1711, 0.2082, 0.2805, 0.2543, 0.1711, 0.2152, 0.169, 0.1374, 0.1782, 0.1864, 0.1061, 0.0863, 0.1758, 0.1121, 0.0776, 0.1544, 0.1088, 0.1845, 0.2555, 0.1052, 0.175, 0.1038, 0.1913, 0.1356, 0.0874, 0.1207, 0.1588, 0.0892, 0.072, 0.171, 0.1258, 0.1249, 0.0467, 0.2277, 0.1325, 0.0835, 0.1526, 0.1092, 0.1178, 0.087, 0.0812, 0.0777, 0.0819]; 
info2.TrainingAccuracy = [2.4, 0.0, 7.199999999999999, 2.4, 4.8, 2.4, 4.8, 7.199999999999999, 9.6, 4.8, 7.199999999999999, 0.0, 7.199999999999999, 2.4, 2.4, 7.199999999999999, 9.6, 14.399999999999999, 4.8, 12.0, 16.8, 7.199999999999999, 14.399999999999999, 2.4, 9.6, 7.199999999999999, 9.6, 12.0, 12.0, 9.6, 19.2, 12.0, 7.199999999999999, 7.199999999999999, 12.0, 12.0, 9.6, 2.4, 14.399999999999999, 12.0, 12.0, 16.8, 12.0, 12.0, 12.0, 7.199999999999999, 9.6, 26.4, 16.8, 14.399999999999999, 12.0, 16.8, 14.399999999999999, 19.2, 16.8, 9.6, 16.8, 21.599999999999998, 19.2, 21.599999999999998, 16.8, 14.399999999999999, 16.8, 14.399999999999999, 19.2, 28.799999999999997, 26.4, 16.8, 14.399999999999999, 19.2, 28.799999999999997, 38.4, 26.4, 26.4, 31.2, 33.6, 28.799999999999997, 26.4, 38.4, 33.6, 33.6, 33.6, 33.6, 36.0, 40.8, 36.0, 45.6, 38.4, 45.6, 45.6, 43.199999999999996, 31.2, 33.6, 48.0, 43.199999999999996, 40.8, 43.199999999999996, 40.8, 38.4, 45.6, 50.4, 45.6, 48.0, 50.4, 50.4, 55.199999999999996, 48.0, 50.4, 50.4, 45.6, 55.199999999999996, 57.599999999999994, 50.4, 52.8, 48.0, 43.199999999999996, 50.4, 60.0, 52.8, 40.8, 40.8, 55.199999999999996, 45.6, 55.199999999999996, 57.599999999999994, 50.4, 57.599999999999994, 50.4, 65.0, 52.8, 67.0, 48.0, 50.4, 67.0, 52.8, 55.199999999999996, 52.8, 57.599999999999994, 52.8, 60.0, 57.599999999999994, 50.4, 52.8, 57.599999999999994, 57.599999999999994, 57.599999999999994, 55.199999999999996, 50.4, 60.0, 57.599999999999994, 55.199999999999996, 60.0, 60.0, 55.199999999999996, 60.0, 60.0, 57.599999999999994, 60.0, 60.0, 57.599999999999994, 57.599999999999994, 55.199999999999996, 55.199999999999996, 60.0, 57.599999999999994, 60.0, 55.199999999999996, 57.599999999999994, 60.0, 60.0, 57.599999999999994, 60.0, 60.0, 55.199999999999996, 57.599999999999994, 60.0, 60.0, 57.599999999999994, 57.599999999999994, 60.0, 57.599999999999994, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0];
%Visualize activation of first convolutional layer
figure1 = figure;
plotTrainingAccuracy_All(info2,numEpochs);
%%
I = deepDreamImage(net1,2,1:20);
figure('Name','Testing: Confusion matrix')
for i = 1:20
    subplot(5,5,i)
    imshow(I(:,:,:,i))
end

act1 = activations(net1,test,14, 'OutputAs','rows');
finalConvtsne = tsne(act1);
figure
gscatter(finalConvtsne(:,1),finalConvtsne(:,2),test.Labels);
title("Testing: Final conv activations");

YTest = classify(net1,test);
test_acc = mean(YTest==test.Labels)
test_confmat=confusionmat(test.Labels,YTest);
figure3 = figure('Name','Testing: Confusion matrix')
heatmap(test_confmat)

YTrain = classify(net1,train);
train_acc = mean(YTrain==train.Labels)
train_confmat=confusionmat(train.Labels,YTrain);
figure4 = figure('Name','Training: Confusion matrix')
heatmap(train_confmat)

YVal = classify(net1,val);
val_acc = mean(YVal==val.Labels)
val_confmat=confusionmat(val.Labels,YVal);
figure5 = figure('Name','Validation: Confusion matrix')
heatmap(val_confmat)

saveas(figure3,'confmat-def-400-1e4-raw.png')

%%
analyzeNetwork(net1)
YTest = classify(net1,test);
test_acc = mean(YTest==test.Labels)
YTest = classify(net1,val);
val_acc = mean(YTest==val.Labels)
YTest = classify(net1,train);
train_acc = mean(YTest==train.Labels)



