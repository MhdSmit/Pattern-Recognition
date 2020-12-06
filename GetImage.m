function [Y] = GetImage(FileName)
X = imread(FileName);
Xt  = X';
Y = [Xt(:,1);Xt(:,2);Xt(:,3);Xt(:,4);Xt(:,5);Xt(:,6);Xt(:,7)];
