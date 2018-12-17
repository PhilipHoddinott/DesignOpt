classdef philipNeuralNet
    %philipNeuralNet Summary of this class goes here
    %   Detailed explanation goes here
    
    properties       
        learningRate;
        Level;
        enableBias;
        actFunSwitch;
    end
    
    methods     
        function obj = philipNeuralNet(inputValues,sizeArr,learningRate,enableBias,actFunSwitch)
            % initalized values
            inputDim = size(inputValues, 1);

            for i =1:length(sizeArr)
                if i==1
                    obj.Level(i).W=rand(sizeArr(i),inputDim);
                    obj.Level(i).W=( obj.Level(i).W)./size( obj.Level(i).W,2);
                else
                    obj.Level(i).W=rand(sizeArr(i),sizeArr(i-1));
                    obj.Level(i).W=( obj.Level(i).W)./size( obj.Level(i).W,2);
                end
                obj.Level(i).z=learningRate*rand(sizeArr(i),1);
                obj.Level(i).A=learningRate*rand(sizeArr(i),1);
                obj.Level(i).dW=learningRate*rand(sizeArr(i),1);
                obj.Level(i).b=learningRate*zeros(sizeArr(i),1);
                obj.Level(i).db=learningRate*obj.Level(i).b;
                
            end
            obj.learningRate=learningRate;
            obj.enableBias=enableBias;
            obj.actFunSwitch=actFunSwitch;
        end
         function funcVal = actFunc(obj,x)
            %actFunc activation function
            %   depending on the activation funciton switch, this performs
            %   sigmoid or TanH
            if obj.actFunSwitch==0 % for sigmoid
                funcVal = 1./(1 + exp(-x)); 
            elseif obj.actFunSwitch==1 % for tanh
                funcVal=tanh(x);
            end
         end
        
        function funcD = dactFunc(obj,x)
            %dactFunc activation func derivative
            %   depending on the activation funciton switch, this performs
            %   derivative of sigmoid or Tanh
            if obj.actFunSwitch==0
                funcD = obj.actFunc(x).*(1 - obj.actFunc(x));
            elseif obj.actFunSwitch==1
                funcD=(1-tanh(x).^2);
            end
        end
        
        function outputVector = netOutput(pNet, inputVector,sizeArr)
            % netOutput get the output of the neural net given an input and 
            %
            % INPUT:
            % pNet : net
            % inputVector : input to net
            % sizeArr : net architecture
            %
            % OUTPUT:
            % outputVector : output of net
            for i=1:length(sizeArr)
                if i==1
                    pNet.Level(i).z=pNet.Level(i).W*inputVector;
                else
                    pNet.Level(i).z=pNet.Level(i).W*pNet.Level(i-1).A;
                end
                pNet.Level(i).A=pNet.actFunc(pNet.Level(i).z);
            end
            outputVector=pNet.Level(i).A;
        end

    end
end