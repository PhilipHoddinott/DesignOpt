function [testAc] = TestNN_func(model,xTest,tTest)
    yp=tTest;
    xp=xTest;
    y_hat = predict(model,xp);
    [M,y_true] = max(yp,[],2);
    testAc=accuracy_score(y_hat,y_true);

    function cache= forward_prop(model,a0)
        
        W1 = model('W1');
        b1=model('b1');
        W2=model('W2');
        b2=model('b2');
        W3= model('W3');
        b3=model('b3');
        
        %{
        W1=nNet.W1;
    W2=nNet.W2;
    W3=nNet.W3;
    b1=nNet.b1;
    b2=nNet.b2;
    b3=nNet.b3;
        %}
        %whos
        %keyboard;
        z1=(a0*W1) +b1;
        a1=tanh(z1);

        
        z2=(a1*W2)+b2;
        a2 = tanh(z2);

        
        z3 = (a2*W3)+b3;

        a3 = softmax(z3);
        
        keySet = {'a0','a1','a2','a3','z1','z2','z3'};
        valueSet = {a0,a1,a2,a3,z1,z2,z3};
        cache = containers.Map(keySet,valueSet,'UniformValues',false);
    end


    function accuracy_scoreVal = accuracy_score(y_hat,y_true)
        accuracy_scoreVal = norm(y_hat-y_true);
        accuracy_scoreVal=100-accuracy_scoreVal;
        accP=0;
        for ik =1:length(y_hat)
            if y_hat(ik,1)==y_true(ik,1)
                accP=accP+1;
            end
        end
        accPct=100*accP/ik;
        accuracy_scoreVal=accPct;


    end
    function y_hat = predict(model,x)
        
        c = forward_prop(model,x);%l
        [Mot,y_hat]=max(c('a3'),[],2);

    end

end

