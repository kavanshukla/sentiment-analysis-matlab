test_data_file = 'test3.txt';

test_data = demo_read_data(test_data_file);

submission_file = 'submission3.txt';


dim = 300;
temp_load_glove = load('mywordmapfile.mat');
wordmap = temp_load_glove.wordmap;
fprintf('Fetched Glove Vector Data \n');

temp_load_weights = load('weights.mat');
W_conv = temp_load_weights.W_conv;
B_conv = temp_load_weights.B_conv;
W_out = temp_load_weights.W_out;
B_out = temp_load_weights.B_out;


fprintf('Fetched Trained Weights \n');


flag = 0;
n_filter = 2;
filter_size = [2,3,4];
eta = 0.1;


total_filters = filter_len * n_filter;
n_class = 2;

pool_res = cell(1, filter_len);
cache = cell(1, filter_len);

fid = fopen(submission_file, 'at');
fprintf(fid, 'id::label \n');
for i=1:length(test_data)

words_array = test_data{i,2};
Sentence_id = test_data{i,1};
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

[~,pred]=max(o);

Label = pred - 1;
fprintf(fid, '%d::%d \n', Sentence_id, Label);
end

fclose(fid);
fprintf('\n Output Saved in text File with format Sentence_ID::Predicted_Label ');
