function [map test_data test_label train_data train_label vocabulary] = LoadDataset()

map = readtable('Dataset/map.csv');
test_data = table2array(readtable('Dataset/test_data.csv','ReadVariableNames',false));
test_label = table2array(readtable('Dataset/test_label.csv','ReadVariableNames',false));
train_data = table2array(readtable('Dataset/train_data.csv','ReadVariableNames',false));
train_label = table2array(readtable('Dataset/train_label.csv','ReadVariableNames',false));

fileID = fopen('Dataset/vocabulary.txt','r');
formatSpec = '%s';
A = textscan(fileID,formatSpec);
fclose(fileID);

vocabulary=A{1,1}(1:end);

clear A
clear fileID
clear formatSpec
clear ans
