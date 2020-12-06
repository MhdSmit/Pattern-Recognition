function [Drawing] = Histogram(X)
H = zeros(10,1);
for i = 1:100,
    for j = 1:10,
        if ((j-1)/10)<X(i) && X(i)<(j/10)
            H(j) = H(j)+1;
        end
    end
end

% Ploting
D0 = ones(1,100);
for j = 1:10,
    Drawing(((j-1)*100+1):j*100) = H(j) * D0;
end
