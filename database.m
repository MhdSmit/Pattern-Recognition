clear all;
%Initialization
load('mnist_all.mat');
D=28*28;
N_test=10;
N_train=10;
arrays = zeros(D,1);
test_arrays= zeros(D,1);
%Number 0 
for i = 1 : N_train
    t0=train0(i,:);
    arrays(:,i) = t0';
    
end  
%Number 1
for i = N_train+1 : 2*N_train
    t0=train1(i-N_train,:);
    arrays(:,i) = t0';
   
end
%Number 2
for i = 2*N_train+1 : 3*N_train
    t0=train2(i-2*N_train,:);
    arrays(:,i) = t0';
        
end
%Number 3
for i = 3*N_train+1 : 4*N_train
    t0=train3(i-3*N_train,:);
    arrays(:,i) = t0';
    
end
%Number 4
for i = 4*N_train+1 : 5*N_train
    t0=train4(i-4*N_train,:);
    arrays(:,i) = t0';
      
end
%Number 5
for i = 5*N_train+1 : 6*N_train
    t0=train5(i-5*N_train,:);
    arrays(:,i) = t0';
     
end
%Number 6
for i = 6*N_train+1 : 7*N_train
    t0=train6(i-6*N_train,:);
    arrays(:,i) = t0';
    
end
%Number 7
for i = 7*N_train+1 : 8*N_train
    t0=train7(i-7*N_train,:);
    arrays(:,i) = t0';
    
end
%Number 8
for i = 8*N_train+1 : 9*N_train
    t0=train8(i-8*N_train,:);
    arrays(:,i) = t0';
  
end
%Number 9
for i = 9*N_train+1 : 10*N_train
    t0=train9(i-9*N_train,:);
    arrays(:,i) = t0';
    
end
%% test
%Number 0
for i = 1 : N_test
    t0=test0(i,:);
    test_arrays(:,i) = t0';
end  
%Number 1
for i = N_test+1 : 2*N_test
    t0=test1(i-N_test,:);
    test_arrays(:,i) = t0';
end
%Number 2
for i = 2*N_test+1 : 3*N_test
    t0=test2(i-2*N_test,:);
    test_arrays(:,i) = t0';
end
%Number 3
for i = 3*N_test+1 : 4*N_test
    t0=test3(i-3*N_test,:);
    test_arrays(:,i) = t0';
end
%Number 4
for i = 4*N_test+1 : 5*N_test
    t0=test4(i-4*N_test,:);
    test_arrays(:,i) = t0';
end
%Number 5
for i = 5*N_test+1 : 6*N_test
    t0=test5(i-5*N_test,:);
    test_arrays(:,i) = t0';
end
%Number 6
for i = 6*N_test+1 : 7*N_test
    t0=test6(i-6*N_test,:);
    test_arrays(:,i) = t0';
end
%Number 7
for i = 7*N_test+1 : 8*N_test
    t0=test7(i-7*N_test,:);
    test_arrays(:,i) = t0';
end
%Number 8
for i = 8*N_test+1 : 9*N_test
    t0=test8(i-8*N_test,:);
    test_arrays(:,i) = t0';
end
%Number 9
for i = 9*N_test+1 : 10*N_test
    t0=test9(i-9*N_test,:);
    test_arrays(:,i) = t0';
end
%% mean value
means = (mean(arrays,2));
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
%% Number of Eigen value required
notError = 0.95;
totalSum = sum(pcaEigenVals);
sumEieg = 0;
num = 0;
for i = 1 : D
    sumEieg = sumEieg + pcaEigenVals(D - i + 1);
    if sumEieg >= notError * totalSum 
        num = i;
        break;
    end
end
num
sumEiegn=zeros(D,1);
for k=1:D
     for i=k+1:D
         sumEiegn(k)=sumEiegn(k)+EigenValues(i);
     end;  
end;
plot( sumEiegn);
%%
K= 60;
X_test=zeros(K,1);
X_train=zeros(K,1);
N_digit=1;
for i=1:10*N_digit
    testImg = zeros(D,1);
    testImg(:,1) =arrays(:,i*N_train/N_digit);
    zeroMeanImg = testImg - means; % substracte mean value from the picture
    projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
    concatedProj = projected .* [zeros(D-K,1);ones(K,1)]; % take the most M significant components
    X_train(:,i)=concatedProj([D-K+1:D],1);
end;

for i=1:1*N_test
    testImg = zeros(D,1);
    testImg(:,1) =test_arrays(:,i);
    zeroMeanImg = testImg - means; % substracte mean value from the picture
    projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
    concatedProj = projected .* [zeros(D-K,1);ones(K,1)]; % take the most M significant components
    X_test(:,i)=concatedProj([D-K+1:D],1);
end;

 e = zeros(D,1); 
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
  plot(e);
%NN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants

LR = 0.2;                           % Learning Rate
E = 0.01;                           % Stop Condition
N = 60;                             % Neurons in The Input Layer 
L = 7;                              % Neurons in The Hidden Layer
M = 10;                             % Neurons in The Output Layer

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
% y=zeros(1,L);
% out=zeros(1,M);
% error=zeros(1,M);
Target = zeros(1,M);
Target_h=diag(ones(10,1));
for j=1:10
    for i = 1 : N_digit
         Target(i+((j-1)*N_digit),:)=Target_h(j,:);
    end;
end;

 Targett= Target';

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
 N_test=10
Target_test = zeros(1,M);
for j=1:10
    for i = 1 : N_test
         Target_test(i+((j-1)*N_test),:)=Target_h(j,:);
    end;
end;
max_T=zeros(1,10*N_digit);
index_T=zeros(1,10*N_digit);
max1=zeros(1,10*N_digit);
index=zeros(1,10*N_digit);
% N_test
num_error=0;
for i=1:10
         [Y_T(i,:),Out_T(i,:)] = Feedforward( X_test(:,i),Wij,Vjk,W0,V0);
%             Out_T(i,:) =net( X_train(:,i)) ;
          [max_T(i),index_T(i)] =max (Target(:,i));
           [max1(i),index(i)] =max ( Out_T(i,:));
         if  index_T(i)~= index(i)
             num_error=num_error+1;
         end;
end;
num_error