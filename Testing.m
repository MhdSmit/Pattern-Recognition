function [] = Testing(X , Wij , Vjk , W0 , V0 , Segma)

for i = 1:100
    Noise = Segma * randn(60,1);
    M(:,i) = X + Noise;
    [Y,Res(i,:)] = Feedforward(M(:,i),Wij,Vjk,W0,V0);
end
t = 0.001:0.001:1;
D = zeros(10,1000);
for i = 1:10
    D(i,:) = Histogram(Res(:,i));
    subplot(5,2,i),plot(t,D(i,:)); axis([0 1 0 105])
end
