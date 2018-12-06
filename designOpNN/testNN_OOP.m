close all; clear all;

x0=1;
x1=2;
y0=3;
y1=4;

net=philipNN(x0,x1,y0,y1)


net.y1=net.y1+net.y0