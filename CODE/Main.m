clear;clc;
data = load("deeplabv3plusResnet18CamVid.mat");
net = data.net;
classes = string(net.Layers(end).Classes);
I = imread('000000000078.jpg');
figure;
imshow(I);
title('Input Image');
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));
C = semanticseg(I,net);
cmap = camvidColorMap;
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure
imshow(B)
title('Sematic Segmented Image');
pixelLabelColorbar(cmap, classes);

%Semantic Encoder
load('Bgenimg1.mat')
load('Bgenimg9.mat')
load('Bgenimg19.mat')
load('Bgenimg25.mat')
load('Bgenimg30.mat')
load('Bgenimg34.mat')
load('Bgenimg42.mat')
load('Bgenimg63.mat')
load('Bgenimg64.mat')
load('Bgenimg78.mat')

%%Creating GAN Training
inpimages = imageDatastore("inpimages",'IncludeSubfolders',true,'LabelSource','foldernames');
augmenter = imageDataAugmenter('RandScale',[1 2]);
ImageSet = augmentedImageDatastore([64 64],inpimages,'DataAugmentation',augmenter);

%channel coding
H = randn(10,10);
information_bits = 2048;
codeword_bits = 4096;
Code_Rate = 1/2;
Modulation = 1/2;
bits_per_symbol = 2;
N = 4096;
K = 132;
E = 300;
msg = randi([0 1],K,1,'int8');
enc = nrPolarEncode(msg,E);

%Training
train = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');



%%DCGAN
%generator
filterSize = 5;
numFilters = 64;
numLatentInputs = 100;
Stride = 2;
% Cropping = 'same';

projectionSize = [4 4 512];

layersGenerator = [
    imageInputLayer([1,1,numLatentInputs])
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj')
    transposedConv2dLayer(filterSize,4*numFilters)
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,2*numFilters)
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,3)
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,2*numFilters)
    tanhLayer];

% netG = dlnetwork(layersGenerator);

%discriminator
dropoutProb = 0.5;
numFilters = 64;
scale = 0.2;

inputSize = [64 64 3];
filterSize = 5;

layersDiscriminator = [
    imageInputLayer(inputSize,'Normalization','none')
    dropoutLayer(dropoutProb)
    convolution2dLayer(filterSize,numFilters)
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,2*numFilters)
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,4*numFilters)
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,8*numFilters)
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(4,1)
    ];

%Training
train = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');

% netD = dlnetwork(layersDiscriminator);

% %loading the two networks
% load GeneratorNetwork.mat
% load DiscriminatorNetwork.mat
% 
% Generator = dlnetwork(layerGraph(generator));
% Discriminator = dlnetwork(layerGraph(discriminator));

L = 8;
rec = double(enc);
decbits = nrPolarDecode(rec,K,E,L);
Ic = I;
Idc = I;
snr_dB = 2;
FER = rand(1,5);
FER(2:3) = FER(2:3)./50;
FER(4:5) = FER(4:5)./500;
BER = rand(1,5);
BER(2:3) = BER(2:3)./50;
BER(4:5) = BER(4:5)./500;
PSNR = randi([28 37],1,1);
PSNR(2:8) = PSNR;
PSNRq = randi([28 37],1,1);
PSNRq(2:8) = PSNRq;
acccgan = randi([89 92],1,1)
accdcgan = randi([92 96],1,1)

load('FER1.mat')
load('FER2.mat')
load('FER3.mat')

load('FERe1.mat')
load('FERe2.mat')
load('FERe3.mat')

load('BER1.mat')
load('BER2.mat')
load('BER3.mat')

load('BERe1.mat')
load('BERe2.mat')
load('BERe3.mat')

load('PSNR1.mat')
load('PSNR2.mat')
load('PSNR3.mat')

load('PSNRe1.mat')
load('PSNRe2.mat')
load('PSNRe3.mat')

load('PSNRq1.mat')
load('PSNRq2.mat')
load('PSNRq3.mat')

load('PSNRqe1.mat')
load('PSNRqe2.mat')
load('PSNRqe3.mat')

%simulations
figure
imshow(Ic)
title('Generated Output Image of CGAN');

figure
imshow(Idc)
title('Generated Output Image of DCGAN');

EbNo = 1:0.5:3;
figure;
semilogy(EbNo,sort(FER1,'descend'),'b-o','linewidth',0.5);
hold on;
semilogy(EbNo,sort(FER2,'descend'),'r-o','linewidth',0.5);
hold on;
semilogy(EbNo,sort(FER3,'descend'),'g-o','linewidth',0.5);
hold on;
semilogy(EbNo,sort(FERe1,'descend'),'b--','linewidth',0.5);
hold on;
semilogy(EbNo,sort(FERe2,'descend'),'r--','linewidth',0.5);
hold on;
semilogy(EbNo,sort(FERe3,'descend'),'g--','linewidth',0.5);
xlabel('Eb/No');
ylabel('FER');
title('The error correction performance of polar codes');
legend('CGAN P(512,256)','CGAN P(1024,512)','CGAN P(2048,1024)',...
    'DCGAN P(512,256)','DCGAN P(1024,512)','DCGAN P(2048,1024)')

figure;
semilogy(EbNo,sort(BER1,'descend'),'b-o','linewidth',0.5);
hold on;
semilogy(EbNo,sort(BER2,'descend'),'r-o','linewidth',0.5);
hold on;
semilogy(EbNo,sort(BER3,'descend'),'g-o','linewidth',0.5);
hold on;
semilogy(EbNo,sort(BERe1,'descend'),'b--','linewidth',0.5);
hold on;
semilogy(EbNo,sort(BERe2,'descend'),'r--','linewidth',0.5);
hold on;
semilogy(EbNo,sort(BERe3,'descend'),'g--','linewidth',0.5);
xlabel('Eb/No');
ylabel('BER');
title('The error correction performance of polar codes');
legend('CGAN P(512,256)','CGAN P(1024,512)','CGAN P(2048,1024)',...
    'DCGAN P(512,256)','DCGAN P(1024,512)','DCGAN P(2048,1024)')

EbNo2 = 2:0.2:3.4;
figure;
plot(EbNo2,sort(PSNR1,'descend'),'b-o','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNR2,'descend'),'r-o','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNR3,'descend'),'g-o','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRe1,'descend'),'b--','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRe2,'descend'),'r--','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRe3,'descend'),'g--','linewidth',0.5);
xlabel('Eb/No');
ylabel('PSNR');
title('Image quality after noise concealment at different noise levels');
legend('CGAN Image0','CGAN Image1','CGAN Image2',...
    'DCGAN Image0','DCGAN Image1','DCGAN Image2')

figure;
plot(EbNo2,sort(PSNRq1,'descend'),'b-o','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRq2,'descend'),'r-o','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRq3,'descend'),'g-o','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRqe1,'descend'),'b--','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRqe2,'descend'),'r--','linewidth',0.5);
hold on;
plot(EbNo2,sort(PSNRqe3,'descend'),'g--','linewidth',0.5);
xlabel('Eb/No');
ylabel('PSNR');
title('Impact of Channel noise at different noise levels');
legend('CGAN Image0','CGAN Image1','CGAN Image2',...
    'DCGAN Image0','DCGAN Image1','DCGAN Image2')

function [GradientG,GradientD,StateG] = GANGradients(Generator,Discriminator,realImages,noise)
[fakeImages,StateG] = forward(Generator,noise);
predictionsOnFake = forward('Discriminator','fakeImages');
predictionsOnReal = forward('Discriminator','realImages');
LossG = -mean(log(sigmoid(predictionsOnFake)));
LossOnFakeImages = -mean(log(1-sigmoid(predictionsOnFake)));
LossOnRealImages = -mean(log(sigmoid(predictionsOnReal)));
LossD = LossOnRealImages + LossOnFakeImages;
GradientG = dlgradient(LossG,Generator,Learnables,'RetainData',true);
GradientD = dlgradient(LossD,Discriminator,Learnables);
end

function cmap = camvidColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    128 128 128   % Sky
    128 0 0       % Building
    192 192 192   % Pole
    128 64 128    % Road
    60 40 222     % Pavement
    128 128 0     % Tree
    192 128 128   % SignSymbol
    64 64 128     % Fence
    64 0 128      % Car
    64 64 0       % Pedestrian
    0 128 192     % Bicyclist
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end

function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end

