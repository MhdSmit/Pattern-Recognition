function [Yout] = Neuron(Xin,Win,W0)
Sum = W0 + Xin * Win;  
Yout = 1/( 1 + exp(-Sum) );