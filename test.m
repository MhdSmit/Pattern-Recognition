

N_test=6;
test_arrays= zeros(D,1);

%% Create an array of filenames that make up the image sequence
dirOutput = dir('C:\Users\Smit\Desktop\Pattern Recognition\Pattern Recognition\pca+NN(1)\Test\*_*.bmp');
fileNames = {dirOutput.name}';
numFrames = numel(fileNames);
I = rgb2gray(imread(fileNames{1}));
imgSize = size(I);
D = imgSize(1) * imgSize(2);
% Preallocate the array
sequence = zeros([imgSize numFrames]);
test_arrays = zeros(D,1);
sequence(:,:,1) = I;
test_arrays(:,1) = reshape(I,[D 1]);
% Create image sequence array
for i = 2 : numFrames
    sequence(:,:,i) = rgb2gray(imread(fileNames{i})); 
  %  imshow(sequence(:,:,57))
    test_arrays(:,i) = reshape(sequence(:,:,i),[D 1]);
end


for i=1:10*N_test
    testImg = zeros(D,1);
    testImg(:,1) =test_arrays(:,i);
    zeroMeanImg = testImg - means; % substracte mean value from the picture
    projected = eigenVects' * zeroMeanImg; % move to the new space: bases are eigen vectors
    concatedProj = projected .* [zeros(D-K,1);ones(K,1)]; % take the most M significant components

    X_test(:,i)=concatedProj([D-K+1:D],1);
end;

%*N_train/N_test-1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
% Run one Of These Tests
% You Can Change Segma
N_test=6
Target_test = zeros(1,M);
for j=1:10
    for i = 1 : N_test
         Target_test(i+((j-1)*N_test),:)=Target_h(j,:);
    end;
end;


num_error=0;
for i=1:10*N_test
%          [Y_T(i,:),Out_T(i,:)] = Feedforward( X_test(:,i),Wij,Vjk,W0,V0);
         Out_T(i,:) =net1( X_test(:,i)) ;
        error_test(i) = (Target_test(i,:)- Out_T(i,:))*(Target_test(i,:)- Out_T(i,:))';
        if  error_test(i)>1
            num_error=num_error+1;
        end;
end;
 Total_testError = sum(error_test);
 i
num_error

X0= X_train(:,29)
Y = net(X0)
Y'

 
i=2
 [Y_Tt,Out_Tt] = Feedforward( X_test(:,i),Wij,Vjk,W0,V0)
  error_testt = (Target_test(i,:)- Out_Tt)*(Target_test(i,:)- Out_Tt)'


 X0= X_test(:,57);
Segma = 0;
Testing(X0,Wij,Vjk,W0,V0,Segma);


