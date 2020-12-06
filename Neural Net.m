% You Can Change The Iteration Value To Save Time
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants

LR = 0.2;                           % Learning Rate
E = 0.01;                           % Stop Condition
N = 56;                             % Neurons in The Input Layer 
L = 7;                              % Neurons in The Hidden Layer
M = 10;                             % Neurons in The Output Layer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial Values

Target = diag(ones(10,1));          % Initial Value of The Desired Output
Iteration = 12000;
TotalError = 1;
B = 0.7 * L^(1/N);                
W0 = ones(1,7);                       % Weights Between The Input Layer & The Hidden Layer
Wij = B*(2*rand(N,L)-1);
V0 = ones(1,10);                      % Weights Between The Hidden Layer & The Output Layer
Vjk = rand(L,M)-0.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training Vectors

X0 = GetImage('0.bmp');
X1 = GetImage('1.bmp');
X2 = GetImage('2.bmp');
X3 = GetImage('3.bmp');
X4 = GetImage('4.bmp');
X5 = GetImage('5.bmp');
X6 = GetImage('6.bmp');
X7 = GetImage('7.bmp');
X8 = GetImage('8.bmp');
X9 = GetImage('9.bmp');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while TotalError > E
    if Iteration ~= 0        
        % x The Input Training Pattern Throughout The Neural Net 
        [Y0,Out0] = Feedforward(X0,Wij,Vjk,W0,V0);
        error(1) = (Target(1,:)- Out0)*(Target(1,:)- Out0)';

        [Y1,Out1] = Feedforward(X1,Wij,Vjk,W0,V0);
        error(2) = (Target(2,:)- Out1)*(Target(2,:)- Out1)';
    
        [Y2,Out2] = Feedforward(X2,Wij,Vjk,W0,V0);
        error(3) = (Target(3,:)- Out2)*(Target(3,:)- Out2)';
 
        [Y3,Out3] = Feedforward(X3,Wij,Vjk,W0,V0);
        error(4) = (Target(4,:)- Out3)*(Target(4,:)- Out3)';

        [Y4,Out4] = Feedforward(X4,Wij,Vjk,W0,V0);
        error(5) = (Target(5,:)- Out4)*(Target(5,:)- Out4)';
    
        [Y5,Out5] = Feedforward(X5,Wij,Vjk,W0,V0);
        error(6) = (Target(6,:)- Out5)*(Target(6,:)- Out5)';
    
        [Y6,Out6] = Feedforward(X6,Wij,Vjk,W0,V0);
        error(7) = (Target(7,:)- Out6)*(Target(7,:)- Out6)';
    
        [Y7,Out7] = Feedforward(X7,Wij,Vjk,W0,V0);
        error(8) = (Target(8,:)- Out7)*(Target(8,:)- Out7)';

        [Y8,Out8] = Feedforward(X8,Wij,Vjk,W0,V0);
        error(9) = (Target(9,:)- Out8)*(Target(9,:)- Out8)';
    
        [Y9,Out9] = Feedforward(X9,Wij,Vjk,W0,V0);
        error(9) = (Target(10,:)- Out9)*(Target(10,:)- Out9)';
        
        TotalError = sum(error);
        
        % Backpropagation Analysis Of The Error  
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(1,:),LR , X0,Y0,Out0 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(2,:),LR , X1,Y1,Out1 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(3,:),LR , X2,Y2,Out2 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(4,:),LR , X3,Y3,Out3 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(5,:),LR , X4,Y4,Out4 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(6,:),LR , X5,Y5,Out5 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(7,:),LR , X6,Y6,Out6 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(8,:),LR , X7,Y7,Out7 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(9,:),LR , X8,Y8,Out8 , Wij,Vjk,W0,V0);
        [Wij , Vjk , W0 ,V0] = Backpropagation(Target(10,:),LR , X9,Y9,Out9 , Wij,Vjk,W0,V0);
        
        Iteration = Iteration - 1;
    else
        break
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
% Run one Of These Tests
% You Can Change Segma

[Yt,Outt] = Feedforward(X9,Wij,Vjk,W0,V0)


Segma = 0.2;
Testing(X0,Wij,Vjk,W0,V0,Segma);
Testing(X1,Wij,Vjk,W0,V0,Segma);
Testing(X2,Wij,Vjk,W0,V0,Segma);
Testing(X3,Wij,Vjk,W0,V0,Segma);
Testing(X4,Wij,Vjk,W0,V0,Segma);
Testing(X5,Wij,Vjk,W0,V0,Segma);
Testing(X6,Wij,Vjk,W0,V0,Segma);
Testing(X7,Wij,Vjk,W0,V0,Segma);
Testing(X8,Wij,Vjk,W0,V0,Segma);
Testing(X9,Wij,Vjk,W0,V0,Segma);
