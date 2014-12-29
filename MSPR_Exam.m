clc, clear all, close all;

%removing prwaitbar to reduce computation time
prwaitbar off 

%open file and load in data to Idat
fid = fopen('Skin_NonSkin.txt');
Idat = textscan(fid, '%f%f%f%s','delimiter',' ');
fclose(fid);

%Seperate data from classifier.
data=[Idat{1} Idat{2} Idat{3}];

%Generate prdata
prdat = prdataset(data, Idat{4}, 'featlab',['r';'g';'b'], 'lablist',['skin   ';'nonskin']);

%Generate training set and data set with a 80-20% ratio.
trainSize = round(length(data)*0.8);
[prtrain, prtest]=gendat(prdat,trainSize);