function [test_data] = demo_read_data(filename)
% CMPT-741 example code for: reading training data and building vocabulary.
% NOTE: reading testing data is similar, but no need to build the vocabulary.
%
% return: 
%       data(cell), 1st column -> sentence id, 2nd column -> words, 3rd column -> label
%       wordMap(Map), contains all words and their index, get word index by calling wordMap(word)

headLine = true;
separater = '::';

words = [];
test_data = cell(1000, 3);

fid = fopen(filename, 'r');
line = fgets(fid);

ind = 1;

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
    
    % save data
    test_data{ind, 1} = sid;
    test_data{ind, 2} = w;
    
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
    
end

fprintf('Finished loading Data \n');