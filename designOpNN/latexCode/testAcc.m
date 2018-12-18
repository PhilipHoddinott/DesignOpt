function [numCorrect, numErrors,acc] = testAcc(pNet, inputValuesTest, labelsTes, sizeArr)
    % testAcc test the accuracy of a net using mnist validation set
    %
    % INPUT:
    % pNet : net
    % inputValuesTest : MNIST Input values for testing
    % labelsTest : MNIST Labels for validating testing
    % sizeArr : net architecture
    %
    % OUTPUT:
    % numCorrect : number of correctly classified numbers.
    % numErrors : number of classification errors.
    % acc : test accuracy
    % 
    numErrors = 0;   numCorrect = 0; % set to zero
    for n = 1: size(inputValuesTest, 2)
        outputVector = pNet.netOutput(inputValuesTest(:, n),sizeArr);
        maxVal = 0; classVal = 1;
        
        for i = 1: size(outputVector, 1)
            if outputVector(i) > maxVal
                maxVal = outputVector(i); % get maxV
                classVal = i; % get class
            end
        end
        
        if classVal == labelsTes(n) + 1
            numCorrect = numCorrect + 1; % count correct
        else
            numErrors = numErrors + 1; % count error
        end
    end
    acc=100*(numCorrect) / (numCorrect+numErrors);   % get accuracy
end