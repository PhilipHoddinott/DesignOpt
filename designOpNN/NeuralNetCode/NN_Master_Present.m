% This will throw an error so you will look here and look at your notes
% below
%abc =valNothere
% see
% C:\Users\Philip\Documents\MATLAB\Fall2018\DesignOpt\checkNN\matlab-mnist-two-layer-perceptron-master\matlab-mnist-two-layer-perceptron-master
%% Philip Hoddinott NN
% Neural Net for MNIST numbers
%% Setup
% Setup Enviorment, get data
% Note this may take a while
clear all; close all;


inputValues = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');



% Transform the labels to correct target values.
targetValues = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetValues(labels(n) + 1, n) = 1;
end;
%{
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

load('I_train.mat');
load('L_T_labels.mat');
xTrain=images';

load('I_test.mat');
xTest=images';

load('L_Tst_labels.mat');
Yvals=tTrain;
Xvals=xTrain;

YvalsM=Yvals;
XvalsM=Xvals;
%}
nn_input_dim=784;
nn_hdim=250;
nn_output_dim=10;

numberOfHiddenUnits=nn_hdim;
learningRate=.1;
chunk=1; % creat esub arrays
%pNet=philipNet(nn_input_dim,nn_hdim,nn_output_dim,learningRate,chunk);
pNet=philipNet2(inputValues,targetValues,numberOfHiddenUnits);


inputValuesTest = loadMNISTImages('t10k-images.idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels.idx1-ubyte');
activationFunction = @ActFunc;
dActivationFunction = @dActFunc;
batchSize = 100;
epochs = 500;
numEpoch=10;
load('wksp_best','pNet')
fprintf('Train twolayer perceptron with %d hidden units.\n', nn_hdim);
fprintf('Learning rate: %d.\n', learningRate);
%[pNet,error] = handleTrainNet(pNet, activationFunction, dActivationFunction,inputValues, targetValues, epochs, batchSize,learningRate)
%[pNet,error] = handleTrainNet2(pNet, activationFunction, dActivationFunction,inputValues, targetValues, epochs, batchSize,learningRate)
[pNet,error]=handleTrainNet(pNet,activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate,inputValuesTest, labelsTest,numEpoch);
%[pNet,error]=handleTrainNet3(pNet,activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate);
fprintf('Validation:\n');

[correctlyClassified, classificationErrors] = testAcc(activationFunction, pNet, inputValuesTest, labelsTest);

fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);
acc=100*(correctlyClassified)/(correctlyClassified+classificationErrors);
fprintf('accuracy: %.5f\n', acc);
save('wkspc_1')
%keyboard

function retV = softmax(z)
    exp_scores = exp(z);
    retV= exp_scores./(sum(exp_scores,2));

end
function loss = softmax_loss(y,y_hat)
    minval = 0.000000000001;
    m=length(y);
    y_hatCp=y_hat;
    y_hatCp(y_hatCp<minval)=minval;

    if ~isempty(y_hatCp)

        if y_hatCp~=y_hat
            fprintf('clip\n');

            y_hat=y_hatCp;
        end
    end
    %keyboard
    loss= -1/m * sum(y.*log(y_hat));

end
function lossDeriv = loss_derivative(y,y_hat)
    lossDeriv=y_hat-y;
end
function tndv = tanh_derivative(x)
    tndv=4./((exp(-x)+exp(x)).^2);
end

function funcD=dActFunc(x)
    % tndv=4./((exp(-x)+exp(x)).^2); % tan h deriv 
     funcD = ActFunc(x).*(1 - ActFunc(x));
end

function funcVal = ActFunc(x)
    funcVal = 1./(1 + exp(-x)); % logistic sigmoid
     %y = 1./(1 + exp(-x));
     %funcVal=y;
end
 

function nNet= forward_prop(nNet,a0)
    W1=nNet.W1;
    W2=nNet.W2;
 
    b1=nNet.b1;
    b2=nNet.b2;

    z1=(a0*W1) +b1;
    a1=ActFunc(z1);
    
    
    z2=(a1*W2)+b2;
    a2 = softmax(z2); 

    nNet.a0=a0;
    nNet.a1=a1;
    nNet.a2=a2;

    nNet.z1=z1;
    nNet.z2=z2;

end

function nNet= backwards_prop(nNet,y)
    W1=nNet.W1;
    W2=nNet.W2;
    
    b1=nNet.b1;
    b2=nNet.b2;
    
    a0=nNet.a0;
    a1=nNet.a1;
    a2=nNet.a2;
    
    
    z1=nNet.z1;
    z2=nNet.z2;
    
    m=length(y);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BIG NOTE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %keyboard;
    %loss_derivative(y,a3);
    dz2 =loss_derivative(y,a2);

    dW2 = (1/m)*((a1')*dz2);
    
    db2 = (1/m)*sum(dz2);

    
    dz1 = ((dz2*(W2')) .*dActFunc(z1));
    
    dW1 = (1/m)*((a0')*dz1);
    
    db1 = (1/m)*sum(dz1);
    
    nNet.dW1=dW1;
    nNet.dW2=dW2;
    
    nNet.db1=db1;
    nNet.db2=db2;
    
end

function nNet = load_parameters(netBest,nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk)
    nNet=philipNet_1(nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk);
    rng('shuffle')
    
    nNet.W1=netBest.W1;
    nNet.W2=netBest.W2;
    
    nNet.b1=netBest.b1;
    nNet.b2=netBest.b2;
    
    
end

function accuracy_scoreVal = accuracy_score(y_hat,y_true)
    accP=0;
    for ik =1:length(y_hat)
        if y_hat(ik,1)==y_true(ik,1)
            accP=accP+1;
        end
    end
    accPct=100*accP/ik;
    accuracy_scoreVal=accPct;

    
end

function nNet = update_parameters(nNet)

    W1=nNet.W1;
    W2=nNet.W2;
    
    b1=nNet.b1;
    b2=nNet.b2;
    
    learning_rate=nNet.learning_rate;

    W1 =W1- learning_rate * nNet.dW1;
    b1 =b1- learning_rate * nNet.db1;
    
    W2 =W2- learning_rate * nNet.dW2;
    b2 =b2- learning_rate * nNet.db2;
    

    nNet.W1=W1;
    nNet.W2=W2;
    
    nNet.b1=b1;
    nNet.b2=b2;
    
end

function y_hat = predict(model,x)
    % Do forward pass
    c = forward_prop(model,x);%l
    [M,y_hat]=max(c.a2,[],2);

end



function [nNet]   = trainSB(nNet,Xp,yp,epoch)
    for i=1:epoch
        nNet= forward_prop(nNet,Xp);
        nNet = backwards_prop(nNet,yp);
        nNet = update_parameters(nNet);
    end
end

function [pNet,error] = handleTrainNet(pNet,activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate,inputValuesTest, labelsTest,numEpoch)

    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    n = zeros(batchSize);

     %h = animatedline('Color','r','lineStyle','-','Marker','*');
     h = animatedline('Color','red','lineStyle','-','Marker','*');
     hA = animatedline('Color','blue','lineStyle','--','Marker','x');
    %figure; hold on;
    accBest=-1;
    errorOld=100;
    figure; hold on;
    for t = 1: numEpoch*epochs
        for k = 1: batchSize
            % Select which input vector to train on.
            n(k) = floor(rand(1)*trainingSetSize + 1);
            
            % Propagate the input vector through the network.
            inputVector = inputValues(:, n(k));
            pNet.hiddenActualInput = pNet.hiddenWeights*inputVector;
            pNet.hiddenOutputVector = activationFunction(pNet.hiddenActualInput);
            pNet.outputActualInput = pNet.outputWeights*pNet.hiddenOutputVector;
            pNet.outputVector = activationFunction(pNet.outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            pNet.outputDelta = dActivationFunction(pNet.outputActualInput).*(pNet.outputVector - targetVector);
            pNet.hiddenDelta = dActivationFunction(pNet.hiddenActualInput).*(pNet.outputWeights'*pNet.outputDelta);
            
            pNet.outputWeights = pNet.outputWeights - learningRate.*pNet.outputDelta*pNet.hiddenOutputVector';
            pNet.hiddenWeights = pNet.hiddenWeights - learningRate.*pNet.hiddenDelta*inputVector';
            %keyboard;
        end
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(pNet.outputWeights*activationFunction(pNet.hiddenWeights*inputVector)) - targetVector, 2);
        end
        error = error/batchSize;
        plot(t,error,'*')
        %addpoints(h,t,error);
        %plot(t, error,'*');
        if error<errorOld
            errorOld=error;
            errorBest=error;
        %if mod(t,100)==0 || t==1
            [correctlyClassified, classificationErrors] = testAcc(activationFunction, pNet, inputValuesTest, labelsTest);
            acc=100*(correctlyClassified)/(correctlyClassified+classificationErrors);
            if acc>accBest
                accBest=acc;
            end
            %addpoints(hA,t,acc/100);
            strT=sprintf('Epoch = %d, Test Accuracy = %.5f, best error= %.4f',t,acc,errorBest);
            title(strT);
            fprintf('%s best acc = %.4f\n',strT,accBest)
            grid on;
        end
        
        %subplot(2,1,1)
        %plot(t, error,'*');
        %grid on;
        %title('Error')
        %subplot(2,1,2)
        %plot(t,acc,'*')
        %addpoints(h,t,acc);
        
        
        drawnow
    end
    
        
end
