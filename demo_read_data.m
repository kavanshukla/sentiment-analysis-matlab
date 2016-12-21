function [test_data] = demo_read_data(filename)


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