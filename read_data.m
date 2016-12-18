function [data,test_data] = read_data()
% CMPT-741 example code for: reading training data and building vocabulary.
% NOTE: reading testing data is similar, but no need to build the vocabulary.
%
% return: 
%       data(cell), 1st column -> sentence id, 2nd column -> words, 3rd column -> label
%       wordMap(Map), contains all words and their index, get word index by calling wordMap(word)

headLine = true;
separater = '::';

words = [];
data = cell(5000, 3);
test_data = cell(1000, 3);

fid = fopen('train.txt', 'r');
line = fgets(fid);

ind = 1;
tind = 1;
count = 1;

while ischar(line)
    if headLine
        line = fgets(fid);
        headLine = false;
    end
    attrs = strsplit(line, separater);
    sid = str2double(attrs{1});
    
    s = attrs{2};
    w = strsplit(s);
    words = [words w];
    
    y = str2double(attrs{3});
    
    if count > 5000
        % save data
        test_data{tind, 1} = sid;
        test_data{tind, 2} = w;
        test_data{tind, 3} = y;
        tind = tind + 1;
    else
       % save data
        data{ind, 1} = sid;
        data{ind, 2} = w;
        data{ind, 3} = y; 
        ind = ind + 1;
    end
    % read next line
    line = fgets(fid);
    
    count = count + 1;
end

fprintf('Finished loading Data \n');