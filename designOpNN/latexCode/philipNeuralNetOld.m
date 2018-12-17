classdef philipNeuralNet
    %philipNeuralNet Summary of this class goes here
    %   Detailed explanation goes here
    
    properties       
        learningRate;
        Level;
    end
    
    methods     
        function obj = philipNeuralNet(inputValues,sizeArr,learningRate)
            % initalized values
            inputDim = size(inputValues, 1);

            for i =1:length(sizeArr)
                if i==1
                    obj.Level(i).Weights=rand(sizeArr(i),inputDim);
                    obj.Level(i).Weights=( obj.Level(i).Weights)./size( obj.Level(i).Weights,2);
                else
                    obj.Level(i).Weights=rand(sizeArr(i),sizeArr(i-1));
                    obj.Level(i).Weights=( obj.Level(i).Weights)./size( obj.Level(i).Weights,2);
                end
                obj.Level(i).Input=learningRate*rand(sizeArr(i),1);
                obj.Level(i).Vec=learningRate*rand(sizeArr(i),1);
                obj.Level(i).Delta=learningRate*rand(sizeArr(i),1);
            end
            obj.learningRate=learningRate;
        end

    end
end