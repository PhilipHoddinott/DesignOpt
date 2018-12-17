function [numCorrect, numErrors] = testAcc(pNet, inputValues, labels, sizeArr)
    % testAcc test the accuracy of a net using mnist validation set
    %
    % INPUT:
    % pNet : net
    % inputValues : MNIST Input values for training
    % labels : MNIST Labels for validation
    % sizeArr : net architecture
    %
    % OUTPUT:
    % numCorrect : number of correctly classified numbers.
    % numErrors : number of classification errors.
    % 
    testSetSize = size(inputValues, 2);
    numErrors = 0;   numCorrect = 0;
    
    for n = 1: testSetSize
        inputVector = inputValues(:, n);        
        outputVector = pNet.netOutput(inputVector,sizeArr);
        max = 0; class = 1;
        
        for i = 1: size(outputVector, 1)
            if outputVector(i) > max
                max = outputVector(i);
                class = i;
            end
        end
        
        if class == labels(n) + 1
            numCorrect = numCorrect + 1;
        else
            numErrors = numErrors + 1;
        end
    end
end