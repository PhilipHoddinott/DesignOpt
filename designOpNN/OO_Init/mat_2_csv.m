clear all; close all;
load('digittrain_dataset.mat')

tTrain_csv=tTrain';
cMat=[];
A=[1,2;3,4;5,6;7,8];
B=A(:)'
A = [1:5;6:10]

B = reshape(A',[],1)'
keyboard
for i=1:length(xTrainImages)
    cMatTog=cell2mat(xTrainImages(i));
    %cMatT=cMatT(:)';
    cMatT=reshape(cMatTog',[],1)';
    %keyboard
    cMat=[cMat;cMatT];
    if mod(i,100)==0
        fprintf('%d of %d\n',i,length(xTrainImages));
    end
end
imageVec=cMat;
%csvFinal=[cMat,tTrain_csv];
%csvwrite('WcsvIm.csv',csvFinal);
save('trainMat2','imageVec','tTrain_csv')
    
    