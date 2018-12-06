clear all; close all;

load('best_9028.mat')
    W1 = model('W1');
    b1=model('b1');
    W2=model('W2');
    b2=model('b2');
    W3= model('W3');
    b3=model('b3');
 nn_output_dim=10;
 nn_hdim=100;
 nn_input_dim=784;
 chunk=10;
 learning_rate=.5;
    nNet = initialize_parameters(nn_input_dim, nn_hdim, nn_output_dim,learning_rate,chunk);
    nNet.W1=W1;
    nNet.W2=W2;
    nNet.W3=W3;
    nNet.b1=b1;
    nNet.b2=b2;
    nNet.b3=b3;
    save('nNet_90_28.mat','nNet');
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
