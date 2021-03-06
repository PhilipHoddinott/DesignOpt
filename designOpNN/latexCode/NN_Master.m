%% Philip Hoddinott NN
% Neural Net for MNIST numbers
%% Setup
clear all; close all;
%% load values
% These functions come from
% http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
inputValues = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

inputValuesTest = loadMNISTImages('t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');

% change labels 
targetValues = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetValues(labels(n) + 1, n) = 1;
end
% traing paramters
sizeArr=[250;10];
learningRate=.1;
batchSize = 100;
epochs = 500; numEpoch=1;
% net switches
enableBias=0; %  0 for off, 1 for on
actFunSwitch =0;  % 0 for sigmoid, 1 for Tanh
% create net
pNet=philipNeuralNet(inputValues,sizeArr,learningRate,enableBias,actFunSwitch);
% train net
[pNet,pNetBest,errorM,testAccM]=handleTrainNet(pNet, inputValues, targetValues, epochs, batchSize, inputValuesTest, labelsTest,numEpoch,sizeArr);
% plot results
plotAcc

function [pNet,pNetBest,errorM,testAccM] = handleTrainNet(pNet,  inputValues, targetValues, epochs, batchSize, inputValuesTest, labelsTest,numEpoch,sizeArr)
    % handleTrainNet function to train net via batch traning
    % Input
    % pNet : net
    % epochs : number of epochs
    % numEpoch : epoch multiplier 
    % batchSize : size of batch
    % inputVector : MNIST Input vector for training
    % targetVector : MNIST Labels for traning validation
    % sizeArr : net architecture
    % inputValuesTest : MNIST Input vector for testing
    % labelsTest : MNIST Labels for testing validation
    % 
    % Output
    % pNet : net
    % pNetBest : pNet with best accuracy
    % errorM : matrix of traning error
    % testAccM : matrix of test accuracy
    trainingSetSize = size(inputValues, 2); % get traning set size
    errorM=[]; testAccM=[]; % init values
  
    n = zeros(batchSize); % init values
    errorBest=100;  accBest=-1;  % init values
    
    figure; hold on; % init figure
    ylabel('Training Error');     xlabel('Epochs');
    tic
    for t = 1: numEpoch*epochs
        for k = 1: batchSize
            % Select input vector to train on.
            n(k) = floor(rand(1)*trainingSetSize + 1);
            % get inputs and targets
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            % forward propogation
            pNet = forwardProp(pNet,sizeArr,inputVector);
            % backwards Propagation
            pNet=backprop(pNet,sizeArr,inputVector,targetVector);
        end % end foor loop
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            outputVector = pNet.netOutput(inputVector,sizeArr);
            error=error+norm(outputVector- targetVector, 2);
            
        end
        error = error/batchSize; errorM=[errorM,error];
        plot(t,error,'*')
        
        if error<errorBest
            pNetBest=pNet; errorBest=error; % get best error 
        end
            
        if mod(t,25)==0 %
            [numCorrect, numErrors,acc] = testAcc(pNet, inputValuesTest, labelsTest,sizeArr);
            if acc>accBest
                accBest=acc; % get best accuracy
            end
            testAccM=[testAccM,acc];
            fprintf('Epoch = %d,error = %.4f, best acc = %.4f\n',t, error,accBest)
            grid on;
            toc % time to run
        end
        drawnow % draw error
    end
        
end

function pNet = forwardProp(pNet,sizeArr,inputVector)
    % forwardProp function to perform forward propgation
    % 
    % Input
    % pNet : net
    % inputVector : MNIST Input vector for training
    % targetVector : MNIST Labels for traning validation
    % sizeArr : net architecture
    % 
    % Output
    % pNet : net
    for i=1:length(sizeArr)
        if i==1 % 1st layer from input
            pNet.Level(i).z=pNet.Level(i).W*inputVector; 
        else % all other layers
            pNet.Level(i).z=pNet.Level(i).W* pNet.Level(i-1).A; 
        end
        pNet.Level(i).A=pNet.actFunc(pNet.Level(i).z+pNet.Level(i).b); % handle bias
    end
end

function pNet=backprop(pNet,sizeArr,inputVector,targetVector) 
    % backprop function to perform backpropgation
    % 
    % Input
    % pNet : net
    % inputVector : MNIST Input vector for training
    % targetVector : MNIST Labels for traning validation
    % sizeArr : net architecture
    % 
    % Output
    % pNet : net
    learningRate=pNet.learningRate; % get leraning rate
    iArr=linspace(length(sizeArr),1,length(sizeArr)); % array to go backwards
    for i=iArr
        if i==length(sizeArr) % derivative from cost at output
            pNet.Level(i).dW=pNet.dactFunc( pNet.Level(i).z).* (pNet.Level(i).A-targetVector);
        else % derivative for other hidden layers
            pNet.Level(i).dW = pNet.dactFunc( pNet.Level(i).z).*(pNet.Level(i+1).W'* pNet.Level(i+1).dW);
        end
        dz=pNet.dactFunc(pNet.Level(i).z); % vector deriv
        pNet.Level(i).db=(1/length(dz))* sum(dz,2); % bias deriv
    end

    for i=iArr
        if i~=1 % adjust weight for all hidden layers not at input
            pNet.Level(i).W= pNet.Level(i).W-learningRate*pNet.Level(i).dW*pNet.Level(i-1).A';
        else % adjust hidden layer weight at input
            pNet.Level(i).W= pNet.Level(i).W-learningRate*pNet.Level(i).dW*inputVector';
        end
        if pNet.enableBias==1 % switch for enable bias
            pNet.Level(i).b=pNet.Level(i).b-learningRate*pNet.Level(i).db;
        end
    end
end
