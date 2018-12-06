%% Philip Hoddinott NN
% Neural Net for MNIST numbers
%% Setup
% Setup Enviorment, get data
% Note this may take a while
clear all; close all;

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

nn_input_dim=inputSize;
nn_hdim=100;
%% possibly 20
%% posibly sigmoid, not tanH
% https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/ 
% LR = .2?
nn_output_dim=10;
%{
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
%}

learning_rate=.07;
chunk=150; % creat esub arrays
%nNet = initialize_parameters(nn_input_dim, nn_hdim, nn_output_dim,learning_rate,chunk);
%load('wkspc88.mat', 'modelBest')
load('nNet_90_28.mat','nNet');
modelBest=nNet;
%load('bst_L20.mat', 'modelBest')
nNet = load_parameters(modelBest,nn_input_dim, nn_hdim, nn_output_dim,learning_rate,chunk);


%nNet=modelBest;
%nNet.learning_rate=learning_rate;

XvalCr={};
YvalCr={};
for i=1:length(XvalsM(:,1))/chunk
    sVal=1+(i-1)*chunk;
    eVal=(i)*chunk;
    XvalCr(i)={XvalsM(sVal:eVal,:)};
    YvalCr(i)={YvalsM(sVal:eVal,:)};
end
%% Check stuff

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
    [modelCurModel]   = trainSB(nNet,Xvals,Yvals);
        
        a3=nNet.a3;
        %size(a3)
        %size(Yvals)
        lossV=sum(softmax_loss(Yvals,a3));
        y_hat = predict(modelCurModel,Xvals);
        [M,y_true] = max(Yvals,[],2);
        ac=accuracy_score(y_hat,y_true);
        [testAc] = TestNN_funcOO(modelCurModel,xTest,tTest);
    if testAc>bestAc
        bestAc=testAc;
        modelBest=modelCurModel;
    end
    if  mod(iCt,incrmenter)==0
        lossM(pltCount,iCy)=lossV;
        
        accM(pltCount,iCy)=ac;

        addpoints(h,xpt(iCt),testAc);
        st=sprintf('Acc = %.2f, best Acc = %.3f',testAc,bestAc);
        title(st)
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
        [testAc] = TestNN_funcOO(modelCurModel,xTest,tTest);
        fprintf('Test Set accuracy = %.3f\n',testAc);
                   
        
        %st=sprintf('ogLR = %.5e, lr = %.5e,, last lr = %.5e',learning_rateOG,learning_rate,testAcOld);
        testAcOld=testAc;
        learning_rateOld=learning_rate;
        %title(st)
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

function nNet= forward_prop(nNet,a0)
    W1=nNet.W1;
    W2=nNet.W2;
    W3=nNet.W3;
    b1=nNet.b1;
    b2=nNet.b2;
    b3=nNet.b3;

    z1=(a0*W1) +b1;
    a1=tanh(z1);
    
    
    z2=(a1*W2)+b2;
    a2 = tanh(z2);
    
    
    z3 = (a2*W3)+b3;
    
    a3 = softmax(z3);

    nNet.a0=a0;
    nNet.a1=a1;
    nNet.a2=a2;
    nNet.a3=a3;

end

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

    m=length(y);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BIG NOTE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dz3 =loss_derivative(y,a3);

    dW3 = (1/m)*((a2')*dz3);
    
    db3 = (1/m)*sum(dz3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HEREER!!
%% a2 replace with z2 and possibly all the others
    dz2 = ((dz3*(W3')) .* tanh_derivative(a2)); % ask teddy numpy
    
    dW2 = (1/m)*((a1')*dz2);
    
    db2 = (1/m)*sum(dz2);
    
    dz1 = ((dz2*(W2')) .*tanh_derivative(a1));
    
    dW1 = (1/m)*((a0')*dz1);
    
    db1 = (1/m)*sum(dz1);
    
    nNet.dW1=dW1;
    nNet.dW2=dW2;
    nNet.dW3=dW3;
    nNet.db1=db1;
    nNet.db2=db2;
    nNet.db3=db3;
end


function nNet = initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk)
    nNet=philipNet(nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk);
    rng('shuffle')
    W1 = 2*randn(nn_input_dim, nn_hdim) - 1;
    
    b1 = zeros(1,nn_hdim);
    
    W2 = 2*randn(nn_hdim, nn_hdim) - 1;
    
    b2 = zeros(1,nn_hdim);
    
    W3 =  2*rand(nn_hdim, nn_output_dim) - 1;
    
    b3 = zeros(1,nn_output_dim);

    nNet.W1=W1;
    nNet.W2=W2;
    nNet.W3=W3;
    nNet.b1=b1;
    nNet.b2=b2;
    nNet.b3=b3;
    
end



function nNet = load_parameters(netBest,nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk)
    nNet=philipNet(nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk);
    rng('shuffle')
    %{
    W1 = 2*randn(nn_input_dim, nn_hdim) - 1;
    
    b1 = zeros(1,nn_hdim);
    
    W2 = 2*randn(nn_hdim, nn_hdim) - 1;
    
    b2 = zeros(1,nn_hdim);
    
    W3 =  2*rand(nn_hdim, nn_output_dim) - 1;
    
    b3 = zeros(1,nn_output_dim);
    %}
    nNet.W1=netBest.W1;
    nNet.W2=netBest.W2;
    nNet.W3=netBest.W3;
    nNet.b1=netBest.b1;
    nNet.b2=netBest.b2;
    nNet.b3=netBest.b3;
    
end
function accuracy_scoreVal = accuracy_score(y_hat,y_true)

   % accuracy_scoreVal = norm(y_hat-y_true);
   % accuracy_scoreVal=100-accuracy_scoreVal;
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
    W3=nNet.W3;
    b1=nNet.b1;
    b2=nNet.b2;
    b3=nNet.b3;
    learning_rate=nNet.learning_rate;

    W1 =W1- learning_rate * nNet.dW1;
    b1 =b1- learning_rate * nNet.db1;
    W2 =W2- learning_rate * nNet.dW2;
    b2 =b2- learning_rate * nNet.db2;
    W3 =W3- learning_rate * nNet.dW3;
    b3 =b3- learning_rate * nNet.db3;

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

end

%{
function cAcc = calc_accuracy(model,x,y)
    m=length(y);
    pred=predict(model,x);
    error = sum(abs(pred-y));
    cAcc = 100*(m-error)/m ;

end
%}
%{
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
%}

%{
function [nNet,cache,grads]   = trainSB(nNet,Xp,yp,learning_rate,epochs,print_loss)
    cache= forward_prop(nNet,Xp);
    grads = backwards_prop(nNet,cache,yp);
    nNet = update_parameters(nNet,grads,learning_rate);

end
%}

function [nNet]   = trainSB(nNet,Xp,yp)
    nNet= forward_prop(nNet,Xp);
    nNet = backwards_prop(nNet,yp);
    nNet = update_parameters(nNet);
end


    


