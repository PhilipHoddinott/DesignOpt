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
    trainingSetSize = size(inputValues, 2); % get traning set size
    errorM=[]; testAccM=[]; % init values
  
    n = zeros(batchSize); % init values
    errorBest=100;  accBest=-1;  % init values
    
    figure; hold on; % init figure
    ylabel('Training Error')
    xlabel('Epochs')
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
            iArr=linspace(length(sizeArr),1,length(sizeArr));
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
            pNetBest=pNet; errorBest=error;
        end
            
        if mod(t,25)==0 %
            [numCorrect, numErrors] = testAcc(pNet, inputValuesTest, labelsTest,sizeArr);
            acc=100*(numCorrect) / (numCorrect+numErrors);
            if acc>accBest
                accBest=acc;
            end
            testAccM=[testAccM,acc];
            fprintf('Epoch = %d,error = %.4f, best acc = %.4f\n',t, error,accBest)
            grid on;
            toc
        end
        drawnow % draw error
    end
        
end

function pNet = forwardProp(pNet,sizeArr,inputVector)
    for i=1:length(sizeArr)
        if i==1
            pNet.Level(i).z=pNet.Level(i).W*inputVector;
        else % output
            pNet.Level(i).z=pNet.Level(i).W* pNet.Level(i-1).A;
        end
        pNet.Level(i).A=pNet.actFunc(pNet.Level(i).z+pNet.Level(i).b);%%%% NEW
    end
end

function pNet=backprop(pNet,sizeArr,inputVector,targetVector) % function to perform backpropgation
    learningRate=pNet.learningRate;
    iArr=linspace(length(sizeArr),1,length(sizeArr));
    for i=iArr
        if i==length(sizeArr) % cost at output
            pNet.Level(i).dW=pNet.dactFunc( pNet.Level(i).z).* (pNet.Level(i).A-targetVector);
        else % hidden
            pNet.Level(i).dW = pNet.dactFunc( pNet.Level(i).z).*(pNet.Level(i+1).W'* pNet.Level(i+1).dW);
        end
        dz=pNet.dactFunc(pNet.Level(i).z);
        pNet.Level(i).db=(1/length(dz))* sum(dz,2); %%% NEW
        %pNet.Level(i).dW=pNet.Level(i).dW*(1/length(pNet.Level(i).dW)); %% NEW
    end

    for i=iArr
        if i~=1 % output
            pNet.Level(i).W= pNet.Level(i).W-learningRate*pNet.Level(i).dW*pNet.Level(i-1).A';
        else % hidden
            pNet.Level(i).W= pNet.Level(i).W-learningRate*pNet.Level(i).dW*inputVector';
        end
        if pNet.enableBias==1 % switch for enable bias
            pNet.Level(i).b=pNet.Level(i).b-learningRate*pNet.Level(i).db; %%% NEW
        end
    end
end
