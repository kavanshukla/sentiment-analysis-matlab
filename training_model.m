%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: Kavan Shukla
% date: date for release this code
%matconvnet-1.0-beta23/matlab/vl_setupnn
clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
% TODO: your code
correctlabel = 0;
[train_data,validation_data] = read_data();
%save('read_data.mat','train_data','validation_data','wordmap');

%load('read_data.mat');

%[wordmap] = test_vector();
fprintf('Fetched read_data\n');
dim = 300;
temp_load_glove = load('mywordmapfile.mat');
wordmap = temp_load_glove.wordmap;

fprintf('fetched Glove Vector Data \n');



flag = 0;
n_filter = 2;
filter_size = [2,3,4];
eta = 0.1;

filter_len = length(filter_size);
W_conv = cell(filter_len,1);
B_conv = cell(filter_len,1);
new_W_conv = cell(filter_len,1);
new_B_conv = cell(filter_len,1);




for i = 1: filter_len
f = filter_size(i);

W_conv{i} = normrnd(0,0.1,[f,dim, 1, n_filter]);
B_conv{i} = zeros(n_filter,1);

end

%fprintf('Convolution network details');
%disp(W_conv{1,1})

total_filters = filter_len * n_filter;
%fprintf('total filters');
%disp(total_filters);

n_class = 2;

W_out = normrnd(0,0.1,[total_filters,n_class]);

B_out = zeros(n_class,1);


%% Section 2: training
% Note:
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions:
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()




% for each example in train.txt do
% section 2.1 forward propagation and compute the loss
% TODO: your code

pool_res = cell(1, filter_len);
cache = cell(1, filter_len);

loss = cell(1, length(train_data));

epoch = 20;


for ep = 1:epoch 

for i=1:length(train_data)
words_array = train_data{i,2};
X = [];
if length(words_array)<total_filters
    flag = total_filters - length(words_array);
end
for j=1:length(words_array)
present_word = char(words_array(j));

if isKey(wordmap,present_word)

X = [X ; wordmap(present_word)];
else
    
random_assigned = normrnd(0,0.1,[1,dim]);
X = [X ; wordmap('<unk>')];
end

end
if flag>0
    padding_vectors = normrnd(0,0.1,[1,dim]);
    for m = 1:flag
        X = [X; padding_vectors];
    end
end

[pool_res,cache] = CNN(X,filter_len,W_conv, B_conv,cache,pool_res);

z1 = vl_nnconcat(pool_res,3);


z = reshape(z1,total_filters,1);

W_out = reshape(W_out,total_filters,1,1,n_class);

o = vl_nnconv(z, W_out,B_out);

y = train_data{i,3};
y = y + 1;

loss{i} = vl_nnloss(o,y);

dzloss = 1;
dzxx = cell(filter_len,1);
dzdo = vl_nnloss(o,y,dzloss);

[dzdx, dzdw, dzdb] = vl_nnconv(z, W_out, B_out, dzdo) ;

dzdx1 = reshape(dzdx,1,1,total_filters);

dzdx2 = vl_nnconcat(pool_res,3,dzdx1);


for k = 1:filter_len
if(k < length(X(:,1)))
sizes = size(cache{1,k});
dzdpool= vl_nnpool(cache{2,k},[sizes(1),1],dzdx2{1,k});

dzrelu = vl_nnrelu(cache{1,k},dzdpool);

[dzxx{k},new_W_conv{k},new_B_conv{k}] = vl_nnconv(X,W_conv{k},B_conv{k},dzrelu);
W_conv{k}= W_conv{k} - new_W_conv{k}*eta;
B_conv{k}= B_conv{k} - new_B_conv{k}*eta;

for j=1:length(words_array)
present_word = char(words_array(j));
if isKey(wordmap,present_word)

    continue_loop=1;
else
present_word = '<unk>';
end
wordmap(present_word) = wordmap(present_word) - eta * dzxx{k}(j, :);

end
end
end

end


fprintf('done with epoch %i \n',ep);
fprintf('done with eta %i \n',eta);

end

fprintf('finish training model\n');
for i=1:length(validation_data)

words_array = validation_data{i,2};
X = [];
if length(words_array)<total_filters
    flag = total_filters - length(words_array);
end
for j=1:length(words_array)
present_word = char(words_array(j));
if isKey(wordmap,present_word)

X = [X ; wordmap(present_word)];
else
random_assigned = normrnd(0,0.1,[1,dim]);
X = [X ; wordmap('<unk>')];
end
end
if flag>0
    padding_vector = normrnd(0,0.1,[1,dim]);
    for m = 1:flag
        X = [X; padding_vector];
    end
end
[pool_res,cache] = CNN(X,filter_len,W_conv, B_conv,cache,pool_res);

z1 = vl_nnconcat(pool_res,3);


z = reshape(z1,total_filters,1);

W_out = reshape(W_out,total_filters,1,1,n_class);

o = vl_nnconv(z, W_out,B_out);

y = validation_data{i,3};
y = y+1;


[~,pred]=max(o);
if pred == y
correctlabel = correctlabel + 1;
end


end
fprintf('\n Accuracy ');

accuracy = (correctlabel/length(validation_data));
fprintf('%i',accuracy*100);


save('weights.mat','W_conv','B_conv','W_out','B_out');

