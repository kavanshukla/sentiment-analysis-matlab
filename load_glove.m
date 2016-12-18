filename = 'glove.6B.300d.txt';

fin= fopen(filename);
line = fgets(fin);

index = 1;
wordmap = containers.Map('KeyType','char','ValueType','any');
while ischar(line)
      eachline = strsplit(line, ' ');
      myword = eachline{1} ;
      each_line_vector = eachline(2:length(eachline));
      vector50 = str2double(each_line_vector);
      wordmap(myword) = vector50;

      line = fgets(fin);
      index = index  + 1;
     
            
end
save('mywordmapfile.mat','wordmap');