classdef philipNetSixLayer
    %PHILIPNN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inputVector;
        hiddenWeights;
        hiddenActualInput;
        hiddenOutputVector;
        
        outputWeights;
        
        outputActualInput;
        outputVector;
        
        outputDelta
        hiddenDelta;
        
        learningRate;
        
        
        
        
        W1;
        b1;
        W2;
        b2;

        
        dW1;
        db1;
        dW2;
        db2;

        
        a0;
        a1;
        a2;
   
        nn_input_dim;
        nn_hdim;
        nn_output_dim;
        learning_rate;
        z1;
        z2;
        %{
        L1In;
        L1Vec;
        L1W;
        L1D;
        
        
        Input;
        Vec;
        Weights;
        Delta;
        %}
        
        Level;

    end
    
    methods
        
        function obj = philipNetSixLayer(inputValues,targetValues,numberOfHiddenUnits,sizeArr,learningRate) 
        
        
    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    obj.hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    obj.outputWeights = outputWeights./size(outputWeights, 2);
    
    
    for i =1:length(sizeArr)
        if i==1
            %obj.Level(i).Weights=learningRate*rand(sizeArr(i),inputDimensions);
            obj.Level(i).Weights=rand(sizeArr(i),inputDimensions);
             obj.Level(i).Weights=( obj.Level(i).Weights)./size( obj.Level(i).Weights,2);
        else
            %obj.Level(i).Weights=learningRate*rand(sizeArr(i),sizeArr(i-1));
            obj.Level(i).Weights=rand(sizeArr(i),sizeArr(i-1));
            obj.Level(i).Weights=( obj.Level(i).Weights)./size( obj.Level(i).Weights,2);
        end
        obj.Level(i).Input=learningRate*rand(sizeArr(i),1);
        obj.Level(i).Vec=learningRate*rand(sizeArr(i),1);
        obj.Level(i).Delta=learningRate*rand(sizeArr(i),1);
    end

        end

    end
end

