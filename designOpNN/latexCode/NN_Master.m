% This will throw an error so you will look here and look at your notes
% below
%abc =valNothere
% see
% C:\Users\Philip\Documents\MATLAB\Fall2018\DesignOpt\checkNN\matlab-mnist-two-layer-perceptron-master\matlab-mnist-two-layer-perceptron-master
%% Philip Hoddinott NN
% Neural Net for MNIST numbers
%% Setup

clear all; close all;

inputValues = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% Transform the labels to correct target values.
targetValues = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetValues(labels(n) + 1, n) = 1;
end

nn_input_dim=784;
nn_hdim=250;
nn_output_dim=10;

numberOfHiddenUnits=nn_hdim;

sizeArr=[250;10];
learningRate=.1;
chunk=1; % creat esub arrays

%pNet=philipNetSixLayer(inputValues,targetValues,numberOfHiddenUnits,sizeArr,learningRate) 
enableBias=0; %  0 for off, 1 for on
actFunSwitch =0;  % 0 for sigmoid, 1 for Tanh
pNet=philipNeuralNet(inputValues,sizeArr,learningRate,enableBias,actFunSwitch)
% http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset

inputValuesTest = loadMNISTImages('t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');

batchSize = 100;
epochs = 500;
numEpoch=4;

fprintf('Train twolayer perceptron with %d hidden units.\n', nn_hdim);
fprintf('Learning rate: %d.\n', learningRate);

%[pNet,pNetBest,errorM,testAccM]=handleTrainNet(pNet, inputValues, targetValues, epochs, batchSize, inputValuesTest, labelsTest,numEpoch,sizeArr);
load('wksp_plots2')
figure; hold on
plot(25.*[(1:1:length(testAccM))],testAccM)
xlabel('Epochs')
ylabel('Test Accuracy (%)')
grid on
%set(gca, 'YScale', 'log')

figure; hold on
plot(25.*[(15:1:length(testAccM))],testAccM(15:end))
xlabel('Epochs')
ylabel('Test Accuracy (%)')
grid on
set(gca, 'YScale', 'log')
function [pNet,pNetBest,errorM,testAccM] = handleTrainNet(pNet,  inputValues, targetValues, epochs, batchSize, inputValuesTest, labelsTest,numEpoch,sizeArr)

    trainingSetSize = size(inputValues, 2);
    errorM=[];
    testAccM=[];
  
    n = zeros(batchSize);
    errorBest=100;
    accBest=-1;

    figure; hold on;
    ylabel('Training Error')
    %xlabel('Neural Net Evaluations')
    xlabel('Epochs')
    tic
    for t = 1: numEpoch*epochs

        for k = 1: batchSize
            
            % Select which input vector to train on.
            n(k) = floor(rand(1)*trainingSetSize + 1);
            % get inputs and targets
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            % Propagate the input vector through the network.
            for i=1:length(sizeArr)
                if i==1
                    pNet.Level(i).z=pNet.Level(i).W*inputVector;
                else % output
                    pNet.Level(i).z=pNet.Level(i).W* pNet.Level(i-1).A;
                end
                pNet.Level(i).A=pNet.actFunc(pNet.Level(i).z+pNet.Level(i).b);%%%% NEW
            end
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
            
        if mod(t,25)==0 %100
            [correctlyClassified, classificationErrors] = testAcc(pNet, inputValuesTest, labelsTest,sizeArr);
            acc=100*(correctlyClassified) / (correctlyClassified+classificationErrors);
            if acc>accBest
                accBest=acc;
            end
            testAccM=[testAccM,acc];
            fprintf('Epoch = %d,error = %.4f, best acc = %.4f\n',t, error,accBest)
            grid on;
            toc
        end
        drawnow
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