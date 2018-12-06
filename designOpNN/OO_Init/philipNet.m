classdef philipNet
    %PHILIPNN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        W1;
        b1;
        W2;
        b2;
        W3;
        b3;
        
        dW1;
        db1;
        dW2;
        db2;
        dW3;
        db3;
        
        a0;
        a1;
        a2;
        a3;      
        nn_input_dim;
        nn_hdim;
        nn_output_dim;
        learning_rate;
        z1;
        z2;
        z3;
        %inputLayer;
        %middleLayer;
        %outLayer;

    end
    
    methods
        %function obj = philipNN(x0,x1,y0,y1)
        %function obj = philipNN(W1,W2,W3,b1,b2,b3,dW1,dW2,dW3,db1,db2,db3,a0,a1,a2,a3) 
        function obj = philipNet(nn_input_dim,nn_hdim,nn_output_dim,learning_rate,chunk) 
        %PHILIPNN Construct an instance of this class
            %   Detailed explanation goes here
            obj.learning_rate=learning_rate;
            obj.nn_input_dim=nn_input_dim;
            obj.nn_hdim=nn_hdim;
            obj.nn_output_dim=nn_output_dim;
            obj.W1=zeros(nn_input_dim,nn_hdim);
obj.W2=zeros(nn_hdim,nn_hdim);
obj.W3=zeros(nn_hdim,nn_output_dim);

obj.dW1=zeros(nn_input_dim,nn_hdim);
obj.dW2=zeros(nn_hdim,nn_hdim);
obj.dW3=zeros(nn_hdim,nn_output_dim);
%{
obj.a0=zeros(nn_output_dim,nn_input_dim);
obj.a1=-1*ones(nn_output_dim,nn_hdim);
obj.a2=ones(nn_output_dim,nn_hdim);
obj.a3=zeros(nn_output_dim,nn_output_dim);
%}
%chunk
obj.a0=zeros(chunk,nn_input_dim);
obj.a1=-1*ones(chunk,nn_hdim);
obj.a2=ones(chunk,nn_hdim);
obj.a3=zeros(chunk,nn_output_dim);


obj.z1=-1*ones(chunk,nn_hdim);
obj.z2=ones(chunk,nn_hdim);
obj.z3=ones(chunk,nn_output_dim);

obj.b1=zeros(1,nn_hdim);
obj.b2=zeros(1,nn_hdim);
obj.b3=zeros(1,nn_output_dim);

obj.db1=-1*ones(1,nn_hdim);
obj.db2=(-.5)*ones(1,nn_hdim);
obj.db3=(-.5)*ones(1,nn_output_dim);
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end

