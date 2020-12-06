function [Wnew , Vnew , W0new , V0new] = Backpropagation(Target , LR , X , Y , Out , W , V , W0 , V0)

for i = 1:length(Target)                     % Error In The Output of The Net
    DeltaV(i) = ( Target(i) - Out(i) ) * Out(i) * (1-Out(i));
end
DV0 =  LR * DeltaV;
DV = LR * Y' * DeltaV;

DeltaV_In = DeltaV * V';
for i = 1:size(V,1)                      % Error In The Output of Inner Layer
    DeltaW(i) = DeltaV_In(i) * Y(i) * (1-Y(i));
end

DW0 =  LR * DeltaW;
DW = LR * X * DeltaW;

W0new = W0 + DW0;
Wnew = W + DW;
V0new = V0 + DV0;
Vnew = V + DV;
