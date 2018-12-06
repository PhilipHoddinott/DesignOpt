%% Check this works
%% Then to Object Oretnedt
clear all; close all;


%% Model notes
% W1= 784 x middle (say midel = 100)
% W2 = middle x middle
% W3 = middle x 10

% a0 = 10 x784
% a1 = 10x 100
% a2 = 10 x 100
% a3 = 10x10
% b1=1x100
% b2 = 1x100
% b3 = 1x10


imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;


load('I_train.mat')
load('L_T_labels.mat')
xTrain=images';

load('I_test.mat')
xTest=images';
load('L_Tst_labels.mat')

whos
%keyboard
Yvals=tTrain;
Xvals=xTrain;
nn_input_dim=784;
nn_hdim=100;
nn_output_dim=10;
W1=zeros(nn_input_dim,nn_hdim);
W2=zeros(nn_hdim,nn_hdim);
W3=zeros(nn_hdim,nn_output_dim);

dW1=zeros(nn_input_dim,nn_hdim);
dW2=zeros(nn_hdim,nn_hdim);
dW3=zeros(nn_hdim,nn_output_dim);

a0=zeros(nn_output_dim,nn_input_dim);
a1=zeros(nn_output_dim,nn_hdim);
a2=zeros(nn_output_dim,nn_hdim);
a3=zeros(nn_output_dim,nn_output_dim);

b1=zeros(1,nn_hdim);
b2=zeros(1,nn_hdim);
b3=zeros(1,nn_output_dim);

db1=zeros(1,nn_hdim);
db2=zeros(1,nn_hdim);
db3=zeros(1,nn_output_dim);

%nNet = philipNN(W1,W2,W3,b1,b2,b3,dW1,dW2,dW3,db1,db2,db3,a0,a1,a2,a3)
%nNet=philipNN(nn_input_dim,nn_hdim,nn_output_dim);
epochs=1;

print_loss=1;
learning_rate=.87;
nNet = initialize_parameters(nn_input_dim, nn_hdim, nn_output_dim,learning_rate);
%load('best_9017','modelBest')
%model=modelBest;
%load('best_70.mat', 'model')
YvalsM=Yvals;
XvalsM=Xvals;

chunk=10; % creat esub arrays
XvalCr={};
YvalCr={};
for i=1:length(XvalsM(:,1))/chunk
    sVal=1+(i-1)*chunk;
    eVal=(i)*chunk;
    XvalCr(i)={XvalsM(sVal:eVal,:)};
    YvalCr(i)={YvalsM(sVal:eVal,:)};
end
%% Check stuff


learning_rate=.87;
learning_rateOG=learning_rate;
iCt=1;

iCy=1;
h = animatedline;
xpt=1:1:length(XvalsM(:,1));
incrmenter=10;
accM=[];
ac=-1;
testAc=-1;
bestAc=-1;
figure(1)
grid on
tic
pltCount=1;
testAcOld=-1;
modelCurModel=nNet;
while iCy<10   || testAc<94
    nNet=modelCurModel;
    Xvals=cell2mat(XvalCr(iCt));
    Yvals=cell2mat(YvalCr(iCt));
    [modelCurModel]   = trainSB(nNet,Xvals,Yvals,learning_rate,epochs,print_loss);
    %{
    model
    cache
    grads
     a3=cache('a3')
      a0=cache('a0')
      a1=cache('a1')
      a2=cache('a2')
          W1 = model('W1');
    b1=model('b1');
    W2=model('W2');
    b2=model('b2');
    W3= model('W3');
    b3=model('b3');
    db1=grads('db1');
    dW1=grads('dW1');
    dW2=grads('dW2');
    db2=grads('db2');
    dW3= grads('dW3');
    db3=grads('db3');
    keyboard
    %}
        a3=nNet.a3;%cache('a3');
        lossV=sum(softmax_loss(Yvals,a3));
        y_hat = predict(modelCurModel,Xvals);
        [M,y_true] = max(Yvals,[],2);
        ac=accuracy_score(y_hat,y_true);
        [testAc] = TestNN_func2(modelCurModel,xTest,tTest);
    if testAc>bestAc
        bestAc=testAc;
        modelBest=modelCurModel;
    end
    if  mod(iCt,incrmenter)==0
        lossM(pltCount,iCy)=lossV;
        
        accM(pltCount,iCy)=ac;
        %addpoints(h,xpt(iCt),accM(pltCount,iCy));
        addpoints(h,xpt(iCt),testAc);
        
        %[testAc] = TestNN_func(model,xTest,tTest);
        drawnow
        
        
        
        fprintf('ls aft %d=%.4e, lr = %.4e',iCt,lossV,learning_rate);

        fprintf('lop %d,tstAc = %.2f, bstTac = %.2f, curAc: %d, ',iCy,testAc,bestAc,ac);
        toc
        pltCount=pltCount+1;
        
        
    end
    
    iCt=iCt+1;
    if iCt>i
        if testAc<testAcOld

        end

        iCt=1;
        pltCount=1;
        fprintf('Back again, ');
        iCy=iCy+1;
        [testAc] = TestNN_func(modelCurModel,xTest,tTest);
        fprintf('Test Set accuracy = %.3f\n',testAc);
                   
        
        st=sprintf('ogLR = %.5e, lr = %.5e,, last lr = %.5e',learning_rateOG,learning_rate,testAcOld);
        testAcOld=testAc;
        learning_rateOld=learning_rate;
        title(st)
        modelCurModel=modelBest;
    end
    
    if testAc>bestAc
        bestAc=testAc;
        modelBest=modelCurModel;
    end
    


end
save('wkspce');

function retV = softmax(z)
    exp_scores = exp(z);
    
    retV= exp_scores./(sum(exp_scores,2));

end
function loss = softmax_loss(y,y_hat)
    minval = 0.000000000001;
    m=length(y);
    y_hatCp=y_hat;
    y_hatCp(y_hatCp<minval)=minval;
    %whos
    %keyboard
    if ~isempty(y_hatCp)
        %size(y_hatCp)
        %size(y_hat)
        %save('ys,mat','y_hatCp','y_hat');
        %whos
        %keyboard
        if y_hatCp~=y_hat
            fprintf('clip\n');
            %pause
            y_hat=y_hatCp;
        end
    end
    loss= -1/m * sum(y.*log(y_hat));
    %whos
end
function lossDeriv = loss_derivative(y,y_hat)
    lossDeriv=y_hat-y;
end
function tndv = tanh_derivative(x)
    %tndv = (1-x.^2);
    tndv=4./((exp(-x)+exp(x)).^2);
end

function nNet= forward_prop(nNet,a0)
    W1=nNet.W1;
    W2=nNet.W2;
    W3=nNet.W3;
    b1=nNet.b1;
    b2=nNet.b2;
    b3=nNet.b3;
    %a0=nNet.a0;
    %{
    W1 = model('W1');
    b1=model('b1');
    W2=model('W2');
    b2=model('b2');
    W3= model('W3');
    b3=model('b3');
    %}

    z1=(a0*W1) +b1;
    a1=tanh(z1);
    
    
    z2=(a1*W2)+b2;
    a2 = tanh(z2);
    
    
    z3 = (a2*W3)+b3;
    
    a3 = softmax(z3);
    %whos
    %keySet = {'a0','a1','a2','a3','z1','z2','z3'};
    %valueSet = {a0,a1,a2,a3,z1,z2,z3};
    %cache = containers.Map(keySet,valueSet,'UniformValues',false);
    nNet.a0=a0;
    nNet.a1=a1;
    nNet.a2=a2;
    nNet.a3=a3;
    %nNet.z1=z1;
    %nNet.z2=z2;
    %nNet.z3=z3;
end

%function [dW1, dW2, dW3, db1,db2,db3]= backwards_prop(y)
%function grads= backwards_prop(model, cache,y)
function nNet= backwards_prop(nNet,y)
    W1=nNet.W1;
    W2=nNet.W2;
    W3=nNet.W3;
    b1=nNet.b1;
    b2=nNet.b2;
    b3=nNet.b3;
    a0=nNet.a0;
    a1=nNet.a1;
    a2=nNet.a2;
    a3=nNet.a3;
    %{
    W1 = model('W1');
    b1=model('b1');
    W2=model('W2');
    b2=model('b2');
    W3= model('W3');
    b3=model('b3');
    a0 = cache('a0');
    a1 = cache('a1');
    a2 = cache('a2');
    a3 = cache('a3');
    %}
    m=length(y);
    
    dz3 =loss_derivative(y,a3);
    %whos
    %dW3 = (1/m)*dot((a2'),dz3);
    dW3 = (1/m)*((a2')*dz3);
    
    db3 = (1/m)*sum(dz3);
    
    %dz2 = (dot(dz3,(W3')) * tanh_derivative(a2)); % ask teddy numpy
    a1111=(dz3*(W3'));
    a1112= tanh_derivative(a2);
    %whos
    dz2 = ((dz3*(W3')) .* tanh_derivative(a2)); % ask teddy numpy
    
    %dW2 = (1/m)*dot((a1'),dz2);
    dW2 = (1/m)*((a1')*dz2);
    
    db2 = (1/m)*sum(dz2);
    
    %dz1 = (dot(dz2,(W2')) *tanh_derivative(a1));
    %dz1 = ((dz2*(W2')) *tanh_derivative(a1));
    dz1 = ((dz2*(W2')) .*tanh_derivative(a1));
    
   % dW1 = (1/m)*dot((a0'),dz1);
    dW1 = (1/m)*((a0')*dz1);
    
    db1 = (1/m)*sum(dz1);
    %keySet = {'dW1','dW2','dW3','db1','db2','db3'};
    %valueSet = {dW1,dW2,dW3,db1,db2,db3};
    %grads = containers.Map(keySet,valueSet,'UniformValues',false);
    nNet.dW1=dW1;
    nNet.dW2=dW2;
    nNet.dW3=dW3;
    nNet.db1=db1;
    nNet.db2=db2;
    nNet.db3=db3;
end


function nNet = initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim,learning_rate)
    nNet=philipNet(nn_input_dim,nn_hdim,nn_output_dim,learning_rate);
    rng('shuffle')
    W1 = 2*randn(nn_input_dim, nn_hdim) - 1;
    
    b1 = zeros(1,nn_hdim);
    
    W2 = 2*randn(nn_hdim, nn_hdim) - 1;
    
    b2 = zeros(1,nn_hdim);
    
    W3 =  2*rand(nn_hdim, nn_output_dim) - 1;
    
    b3 = zeros(1,nn_output_dim);
    %whos
    %keyboard;
    %keySet = {'W1','W2','W3','b1','b2','b3'};
    %valueSet = {W1,W2,W3,b1,b2,b3};
    %model = containers.Map(keySet,valueSet,'UniformValues',false);
    nNet.W1=W1;
    nNet.W2=W2;
    nNet.W3=W3;
    nNet.b1=b1;
    nNet.b2=b2;
    nNet.b3=b3;
end
function accuracy_scoreVal = accuracy_score(y_hat,y_true)
    %P = py.sys.path; % ensure Python is in the path
    %PyAC = py.python_get_val.getAcc()
    %accuracy_scoreVal = str2num(string(PyAC));
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
    %whos
    %keyboard
    %add one to all lables
    
    %guess == true
    %results is this
    
end
%function nNet = update_parameters(nNet,learning_rate)
function nNet = update_parameters(nNet)
    %whos
    %keyboard
    %{
    W1 = model('W1');
    b1=model('b1');
    W2=model('W2');
    b2=model('b2');
    W3= model('W3');
    b3=model('b3');
    %}
    W1=nNet.W1;
    W2=nNet.W2;
    W3=nNet.W3;
    b1=nNet.b1;
    b2=nNet.b2;
    b3=nNet.b3;
    learning_rate=nNet.learning_rate;
    %{
    W1 =W1- learning_rate * grads('dW1');
    b1 =b1- learning_rate * grads('db1');
    W2 =W2- learning_rate * grads('dW2');
    b2 =b2- learning_rate * grads('db2');
    W3 =W3- learning_rate * grads('dW3');
    b3 =b3- learning_rate * grads('db3');
    %}
    W1 =W1- learning_rate * nNet.dW1;
    b1 =b1- learning_rate * nNet.db1;
    W2 =W2- learning_rate * nNet.dW2;
    b2 =b2- learning_rate * nNet.db2;
    W3 =W3- learning_rate * nNet.dW3;
    b3 =b3- learning_rate * nNet.db3;
    %keySet = {'W1','W2','W3','b1','b2','b3'};
    %valueSet = {W1,W2,W3,b1,b2,b3};
    %model = containers.Map(keySet,valueSet,'UniformValues',false);
    nNet.W1=W1;
    nNet.W2=W2;
    nNet.W3=W3;
    nNet.b1=b1;
    nNet.b2=b2;
    nNet.b3=b3;
end

function y_hat = predict(model,x)
    % Do forward pass
    c = forward_prop(model,x);%l
    [M,y_hat]=max(c.a3,[],2);
    %whos
    %keyboard
end


function cAcc = calc_accuracy(model,x,y)
    m=length(y);
    pred=predict(model,x);
    error = sum(abs(pred-y));
    cAcc = 100*(m-error)/m ;
    %whos
    %keyboard
end

function [model,losses,yTall,yHall,accM,modelBest,acBest] = trainM(model,Xp,yp,learning_rate,epochs,print_loss)
    h = animatedline;
    incrmenter=50;
    %incrmenter=1;
    x=1:incrmenter:epochs;
    xc=1;
    losses=[];
    yTall=[];
    yHall=[];
    accM=[];
    acOld=-1;
    y_hat = predict(model,Xp);
    [M,y_true] = max(yp,[],2);
    acBest=accuracy_score(y_hat,y_true);
    modelBest=model;
    for i =1:epochs
        modelOld=model;
        cache= forward_prop(model,Xp);
        grads = backwards_prop(model,cache,yp);
        model = update_parameters(model,grads,learning_rate);
        %y_hat = predict(model,Xp);
        %[M,y_true] = max(yp,[],2);
        %ac=accuracy_score(y_hat,y_true);
        %if ac>acBest
        %    modelBest=model;
        %    acBest=ac;
        %end
        
        if print_loss ==1 && mod(i,incrmenter)==0
            a3=cache('a3');
            %whos
            %keyboard
            lossV=sum(softmax_loss(yp,a3));
            fprintf('loss after iteration %d : %.4e,   ',i,lossV);
            y_hat = predict(model,Xp);
            [M,y_true] = max(yp,[],2);
            ac=accuracy_score(y_hat,y_true);
            %cAcc = calc_accuracy(model,Xp,yp)
            fprintf('Accuracy after iteration %d : %.3f\n', i,ac);
            %cAcc = calc_accuracy(model,Xp,yp)
            %fprintf('C acc for %d = %.3f\n',i,cAcc);
            losses=[losses;ac];
            yTall=[yTall;y_true'];
            yHall=[yHall;y_hat'];
            accM=[accM;accuracy_score(y_hat,y_true)];
            addpoints(h,x(xc),accM(xc));
            xc=xc+1;
            drawnow
            %whos
            %keyboard
            if ac>acBest
                modelBest=model;
                acBest=ac;
            end
        
        end
        %if i>200
        %    if accM(xc-1)<accM(xc-2)
        %        model=modelOld;
        %    end
        %end
    end
end


%{
function [nNet,cache,grads]   = trainSB(nNet,Xp,yp,learning_rate,epochs,print_loss)
    cache= forward_prop(nNet,Xp);
    grads = backwards_prop(nNet,cache,yp);
    nNet = update_parameters(nNet,grads,learning_rate);

end
%}

function [nNet]   = trainSB(nNet,Xp,yp,learning_rate,epochs,print_loss)
    nNet= forward_prop(nNet,Xp);
    nNet = backwards_prop(nNet,yp);
    nNet = update_parameters(nNet);

end


    


