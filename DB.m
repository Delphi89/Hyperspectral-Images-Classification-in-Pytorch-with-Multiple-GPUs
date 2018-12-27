% split existing Pavia University Hyperspectral image 
% in training and test lots keeping 24 nearest neighbours 
% for each pixel

% start timer
tic

% load the original databases of Pavia University 
%   from http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
%   and copy those 2 files (paviaU_gt.mat and paviaU.mat) in your current
%   work folder
dataStructure1 = load('paviaU_gt.mat');
dataStructure2 = load('paviaU.mat');

% training lot size
tr_size = 200;

%load training data
[m,n,o] = size(dataStructure2.paviaU);
x = gpuArray(dataStructure1.paviaU_gt);
y = gpuArray(dataStructure2.paviaU);
fprintf('\n 1. original Pavia University database loaded from disk...')
      

% adding  two "0" lines around the original image by creating a large image 
% of "0" and adding the original image in the middle        
db_pavia = gpuArray(zeros(614,344,103)); 
db_pavia_gt = gpuArray(zeros(614,344,1)); 
[m,n,o] = size(db_pavia);
db_pavia_gt(3:m-2,3:n-2,1)=x;
db_pavia(3:m-2,3:n-2,:)=y;
fprintf('\n 2. 2x zero padding added to database...')


% adding all the elements different from 0 in test_data and test_target
i = 1;
j = 1;
count = 0;
test_target = gpuArray(zeros(42776,1));
test_data = gpuArray(zeros(42776,5,5,103));

while i<m
    while j<n
        if db_pavia_gt(i,j,1) ~= 0
            count = count + 1;
            test_target(count,1)   = db_pavia_gt(i,j,1);   
            test_data(count,:,:,:) = db_pavia(i-2:i+2,j-2:j+2,:);               
        end
        j=j+1;

    end
    i=i+1;
    j=1;
end
fprintf('\n 3. test_data and test_target created...')


% splitting training and test
for i = 1 : 9
    xcount = sum(test_target(:) == i);
    temp = floor(xcount / tr_size);
    bag(i) = temp;
end

count1 = 0;
count2 = 0;

train_target = gpuArray(zeros(1800,1));
train_data = gpuArray(zeros(1800,5,5,103));

% create the training data
for i = 1 : 9
    [l,k] = size(test_data);
    for j = 1 : (l-(tr_size*9))
        if test_target(j,1) == i
            count1 = count1 + 1;
            if (mod(count1,(bag(i))) == 0) && (count2 < tr_size*i)
                count2 = count2 + 1;
                train_target(count2,1) = i;
                test_target(j) = []; 
                train_data(count2,:,:,:) = test_data(j,:,:,:);
                test_data(j,:,:,:) = [];              
                count1 = count1 - 1; 
            end
        end
    end
    count1 = 0;
end
fprintf('\n 4. train_data and train_target created...')


% check sizes aof all new vectors
[a1,a2] = size(test_target);
[b1,b2, b3, b4] = size(test_data);
[c1,c2] = size(train_target);
[d1,d2, d3, d4] = size(train_data);

count = zeros(3,9);
rcount = zeros(3,9);

% how many pixels from every class can be found in test lot
for i = 1 : 9
    for j = 1 : a1
        if test_target(j) == i
            count(1,i) = count(1,i)+1;
        end
    end
end

% how many pixels from every class can be found in training lot
for i = 1 : 9
    for j = 1 : c1
        if train_target(j) == i
            rcount(1,i) = rcount(1,i)+1;
        end
    end
end


test_data = gather(test_data);
test_target = gather(test_target);
train_data = gather(train_data);
train_target = gather(train_target);

% save test and training vectors in a new file
save('database200.mat', 'train_data', 'train_target', 'test_data', 'test_target');
fprintf('\n 5. new/clean database saved.')


% stop timer
toc
