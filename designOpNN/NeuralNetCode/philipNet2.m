classdef philipNet2
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

        %inputLayer;
        %middleLayer;
        %outLayer;

    end
    
    methods
        %function obj = philipNet2(nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk) 
        function obj = philipNet2(inputValues,targetValues,numberOfHiddenUnits) 
        
        
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
        %PHILIPNN Construct an instance of this class
            %   Detailed explanation goes here
            %{
            obj.learning_rate=learning_rate;
            obj.nn_input_dim=nn_input_dim;
            obj.nn_hdim=nn_hdim;
            obj.nn_output_dim=nn_output_dim;

            obj.W1=2*randn(nn_input_dim, nn_hdim) - 1;
            obj.W2= 2*randn(nn_hdim, nn_output_dim) - 1;
            obj.b1=zeros(1,nn_hdim);
            obj.b2=zeros(1,nn_output_dim);

            obj.dW1=obj.W1;
            obj.dW2=obj.W2;

            obj.a0=zeros(chunk,nn_input_dim);
            obj.a1=-1*ones(chunk,nn_hdim);
            obj.a2=zeros(chunk,nn_output_dim);


            obj.z1=-1*ones(chunk,nn_hdim);
            obj.z2=ones(chunk,nn_output_dim);



            obj.db1=-1*ones(1,nn_hdim);
            obj.db2=(-.5)*ones(1,nn_output_dim);
            %}
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end

