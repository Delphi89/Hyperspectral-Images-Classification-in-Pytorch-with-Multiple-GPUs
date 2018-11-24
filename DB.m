
% database for generating training (200 pixels in each class) and test data


test_target = pavia_db.test_target;
test_data = pavia_db.test_data;
train_target = pavia_db.train_target;
train_data = pavia_db.train_data;

[a1,a2] = size(test_target)
[b1,b2] = size(test_data)
[c1,c2] = size(train_target)
[d1,d2] = size(train_data)


a3 = round (a1 / 6,0) 

count = zeros(3,9);
rcount = zeros(3,9);

for i = 1 : 9
    for j = 1 : a1
        if test_target(j) == i
            count(1,i) = count(1,i)+1;
        end
    end
end

for i = 1 : 9
    for j = 1 : c1
        if train_target(j) == i
            rcount(1,i) = rcount(1,i)+1;
        end
    end
end

rcount;

for i = 1 : 9 
    count(2,i) = round (count(1,i) / 6,0)
end

count

xc = 1;
xv = 0;

for i = 1 : 9
    for j = 1 : (a1-count(2,i))
        if test_target(j,1) == i 
            train_target((c1+xv),1)= i;
            test_target((c1+xv),:) = [];
            train_data((c1+xv),:)= test_data(j,:);
            test_data((c1+xv),:) = [];
            xc = xc + 1;
            xv = xv + 1;
            %middle((i-1)*count(2,i)+xc) = j;
            j=j-1;
        end

        if xc == (count(2,i))
            xc = 0;
            break
        end
    end
end


[a1,a2] = size(test_target)
[b1,b2] = size(test_data)
[c1,c2] = size(train_target)
[d1,d2] = size(train_data)


for i = 1 : 9
    for j = 1 : a1
        if test_target(j) == i
            count(3,i) = count(3,i)+1;
        end
    end
end

count

for i = 1 : 9
    for j = 1 : c1
        if train_target(j) == i
            rcount(3,i) = rcount(3,i)+1;
        end
    end
end

rcount

save('test_target_20p.mat', 'test_target');
save('test_data_20p.mat', 'test_data');
save('train_target_20p.mat', 'train_target');
save('train_data_20p.mat', 'train_data');
