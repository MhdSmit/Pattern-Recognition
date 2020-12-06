clear all;
%% Create an array of filenames that make up the image sequence
dirOutput = dir('C:\Users\Smit\Desktop\Pattern Recognition\Pattern Recognition\pca+NN(1)\train\*_*.bmp');
fileNames = {dirOutput.name}';
numFrames = numel(fileNames);
I = rgb2gray(imread(fileNames{1}));
imgSize = size(I);
D = imgSize(1) * imgSize(2);
% Preallocate the array
sequence = zeros([imgSize numFrames]);
arrays = zeros(D,1);
sequence(:,:,1) = I;
arrays(:,1) = reshape(I,[D 1]);
% Create image sequence array
for i = 2 : numFrames
    sequence(:,:,i) = rgb2gray(imread(fileNames{i})); 
    %imshow(sequence(:,:,90))
    arrays(:,i) = reshape(sequence(:,:,i),[D 1]);
end
%% mean value
means = (mean(arrays,2));
% imshow(reshape(means,[imgSize(1),imgSize(2)]),[0 255]);
%% covariance matrix: eigen values & eigen vectors
c = cov(arrays');
[eigenVects lambdas] = eig(c);
pcaEigenVals = diag(lambdas); 
figure;
plot(pcaEigenVals,'.');
EigenVectors = fliplr(eigenVects);
EigenValues = fliplr(diag(lambdas)')';
figure;
plot(EigenValues,'.');

notError = 0.95;
totalSum = sum(pcaEigenVals);
sumEieg = 0;
num = 0;
for i = 1 : D
    sumEieg = sumEieg + pcaEigenVals(D - i + 1);
    if sumEieg >= notError * totalSum 
        num = i
        break;
    end
end
sumEiegn=zeros(D,1);
for k=1:D
     for i=k+1:D
         sumEiegn(k)=sumEiegn(k)+EigenValues(i);
     end;  
end;
plot( sumEiegn/max(sumEiegn),'r')
hold on
axis ([0 200 0 1]) 
e = zeros(D,1);
N_digit=1
N_train=10
 for M = 1:150,
      for i = 1:10*N_digit,
M
testImg = zeros(D,1);
testImg(:,1) =arrays(:,i*N_train/N_digit);
zeroMeanImg = testImg - means; % substracte mean value from the picture
projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
concatedProj = projected .* [zeros(D-M,1);ones(M,1)]; % take the most M significant components
aprox = inv(eigenVects') * concatedProj + means; % return to the original spae

e(M)=e(M)+0.5*(aprox-testImg(:,1))'*(aprox-testImg(:,1));
      end;
 end;
plot(e/max(e));
 
%%
K= 60;
X_test=zeros(K,1);
X_train=zeros(K,1);

N_train=10;
N_digit=10;

for i=1:10*N_digit
    testImg = zeros(D,1);
    testImg(:,1) =arrays(:,i*N_train/N_digit);
    zeroMeanImg = testImg - means; % substracte mean value from the picture
    projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
    concatedProj = projected .* [zeros(D-K,1);ones(K,1)]; % take the most M significant components

    X_train(:,i)=concatedProj([D-K+1:D],1);
end;
dirOutput1 = dir('C:\Users\Smit\Desktop\Pattern Recognition\Pattern Recognition\pca+NN(1)\test\*_*.bmp');
fileNames1 = {dirOutput1.name}';
numFrames1 = numel(fileNames1);
I1 = rgb2gray(imread(cat(2, 'test\', fileNames1{1})));
imgSize1 = size(I1);
D1 = imgSize1(1) * imgSize1(2);
% Preallocate the array
sequence1 = zeros([imgSize1 numFrames1]);
test_arrays = zeros(D,1);
sequence1(:,:,1) = I1;
test_arrays(:,1) = reshape(I1,[D1 1]);
for i = 2 : numFrames1
    sequence1(:,:,i) = rgb2gray(imread(cat(2, 'test\',fileNames1{i}))); 
    %imshow(sequence(:,:,90))
    test_arrays(:,i) = reshape(sequence1(:,:,i),[D1 1]);
end
for i=1:1*N_test
    testImg = zeros(D,1);
    testImg(:,1) =test_arrays(:,i);
    zeroMeanImg = testImg - means; % substracte mean value from the picture
    projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
    concatedProj = projected .* [zeros(D-K,1);ones(K,1)]; % take the most M significant components
    X_test(:,i)=concatedProj([D-K+1:D],1);
end;
%NN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants

LR = 0.2;                           % Learning Rate
E = 0.01;                           % Stop Condition
N = 60;                             % Neurons in The Input Layer 
L = 25;                              % Neurons in The Hidden Layer
M = 10;                             % Neurons in The Output Layer
N_digit=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial Values
Iteration = 12000;
TotalError = 1;
B = 0.7 * L^(1/N);                
W0 = ones(1,L);                       % Weights Between The Input Layer & The Hidden Layer
Wij = B*(2*rand(N,L)-1);
V0 = ones(1,10);                      % Weights Between The Hidden Layer & The Output Layer
Vjk = rand(L,M)-0.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Target = zeros(1,M);
Target_h=diag(ones(10,1));
for j=1:10
    for i = 1 : N_digit
         Target(i+((j-1)*N_digit),:)=Target_h(j,:);
    end;
end;
Targett=Target';
run=1;
Total_testError1 =0;
Total_testError2=0;
while TotalError > E
    if Iteration ~= 0    
        % x The Input Training Pattern Throughout The Neural Net 
         for i=1:10*N_digit
              [Y(i,:),Out(i,:)] = Feedforward( X_train(:,i),Wij,Vjk,W0,V0);
              error(i) = (Target(i,:)- Out(i,:))*(Target(i,:)- Out(i,:))';
        end;
        TotalError = sum(error);
        
        % Backpropagation Analysis Of The Error  
        for i=1:10*N_digit
             [Wij , Vjk , W0 ,V0] = Backpropagation(Target(i,:), LR ,  X_train(:,i),Y(i,:),Out(i,:) , Wij,Vjk,W0,V0);
        end;    
        Iteration = Iteration - 1
    else
        break
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Testing
N_test=10;
Target_test = zeros(1,M);
for j=1:10
    for i = 1 : N_test
         Target_test(i+((j-1)*N_test),:)=Target_h(j,:);
    end;
end;

num_error=0;
for i=1:10
        [Y_T(i,:),Out_T(i,:)] = Feedforward( X_test(:,i),Wij,Vjk,W0,V0);
        error_test(i) = (Target_test(i,:)- Out_T(i,:))*(Target_test(i,:)- Out_T(i,:))';
        if  error_test(i)>2 
            num_error=num_error+1;
        end;
end;
 Total_testError = sum(error_test);
num_error

% [Y_Tt,Out_Tt] = Feedforward( X_test(:,150),Wij,Vjk,W0,V0) 
% 
%  X0= X_train(:,1)
% Segma = 0.8;
% Testing(X0,Wij,Vjk,W0,V0,Segma);
% X0= X_train(:,29);
% Y = net(X0);
% Y'
%% projection
M= 56;
imgGray= rgb2gray(imread('C:\Users\Smit\Desktop\Pattern Recognition\Pattern Recognition\pca+NN(1)\9_1.bmp'));
% imgGray= rgb2gray(imread(fileNames{1}));
testImg = zeros(D,1);
testImg(:,1) = reshape(imgGray,[D 1]);
zeroMeanImg = testImg - means; % substracte mean value from the picture
projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
concatedProj = projected .* [zeros(D-M,1);ones(M,1)]; % take the most M significant components

X0=zeros(56,1);
X0=concatedProj([D-M+1:D],1)
X1=zeros(56,1);
X1=concatedProj([D-M+1:D],1)
X2=zeros(56,1);
X2=concatedProj([D-M+1:D],1)
X3=zeros(56,1);
X3=concatedProj([D-M+1:D],1)
X4=zeros(56,1);
X4=concatedProj([D-M+1:D],1)
X5=zeros(56,1);
X5=concatedProj([D-M+1:D],1)
X6=zeros(56,1);
X6=concatedProj([D-M+1:D],1)
X7=zeros(56,1);
X7=concatedProj([D-M+1:D],1)
X8=zeros(56,1);
X8=concatedProj([D-M+1:D],1)
X9=zeros(56,1);
X9=concatedProj([D-M+1:D],1)
X10=zeros(56,1);
X10=concatedProj([D-M+1:D],1)
 


aprox = inv(eigenVects') * concatedProj + means; % return to the original spae
%   e(M)=e(M)+0.5*(aprox-testImg(:,1))'*(aprox-testImg(:,1));
%      end
%  plot(e);

imshow(reshape(aprox',[imgSize(1),imgSize(2)]),[0 255]); % show the result
title('Estimated Image');
figure
imshow(imgGray,[0 255]); % compare with the original
title('Orginal Image');
%%
% imshow(reshape(arrays(:,150),[imgSize(1),imgSize(2)]),[0 255]);




e = zeros(D,1);
for M = 1:200,
  %  for i = 1:30,
    % M= 100;
    %t0=test0(1,:);
M=100
i=200;
testImg = zeros(D,1);
testImg(:,1) =arrays(:,i);
zeroMeanImg = testImg - means; % substracte mean value from the picture
projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
concatedProj = projected .* [zeros(D-M,1);ones(M,1)]; % take the most M significant components
aprox = inv(eigenVects') * concatedProj + means; % return to the original spae

% eig2=eigenVects(:,[D-M+1:D]);
% projected2 = eig2' * zeroMeanImg;
% projected2=flipud(projected2);
% projected2(M+1:D,1) = zeros(D-M,1);
% projected2=flipud(projected2);
% aprox2 = inv(eigenVects') * projected2 + means; % return to the original spae

e(M)=e(M)+0.5*(aprox-testImg(:,1))'*(aprox-testImg(:,1));

%end
end;

 plot(e,'.');

imshow(reshape(aprox',[28,28]),[0 255]); % show the result
title('Estimated Image');
figure
imshow(reshape(t0',[28,28]),[0 255]); % compare with the original
title('Orginal Image');
    

 

