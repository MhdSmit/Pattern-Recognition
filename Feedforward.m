function [Y , Out] = Feedforward(X , Wij , Vij , W0 , V0)

%Preparing Data

N = length(X);                             % Neurons in The Input Layer 
L = size(Wij,2);                              % Neurons in The Hidden Layer
M = length(V0);                             % Neurons in The Output Layer

% Z(1*56) Output Of The Input Layer

for i = 1:N
%     Z(i) = Neuron(X(i),1,0);
      Z(i) = X(i);
end

% Y(1*7) Output Of The Hidden Layer

for i = 1:L
    Y(i) = Neuron(Z,Wij(:,i),W0(i));
end

% Out(1*10) Output Of The Net

for i = 1:M
    Out(i) = Neuron(Y,Vij(:,i),V0(i));
end
