clc, clear all, close all;

%removing prwaitbar to reduce computation time
prwaitbar off 

%open file and load in data to Idat
fid = fopen('Skin_NonSkin.txt');
Idat = textscan(fid, '%f%f%f%s','delimiter',' ');
fclose(fid);

%Seperate data from classifier.
data=[Idat{1} Idat{2} Idat{3}];

% Generate prdata
prdat = prdataset(data, Idat{4}, 'featlab',['red  ';'green';'blue '], 'lablist',['skin   ';'nonskin']);

% Generate training set and data set with a 80-20% ratio.
trainSize = round(length(data)*0.8);
[prtrain, prtest]=gendat(prdat,trainSize);

%% Plot components in a scatter plot against each other. 
colors = {'red' 'green' 'blue'};
for i=2:3,
    for j=1:i-1,
        figure
        scatterd(prtrain(:,[i j]),'legend')
        title(sprintf('Feature %s vs. %s',colors{i},colors{j}))
    end
end
% Skin (blue +) is more grouped together in all three scenarios then
% nonskin (red *). Tentative analysis would suggest that blue (+) is
% skin and the red (*) is nonskin. The red (*) data is more spread out of
% some clusters seem random whereas blue (+) is grouped. 

%% Correlation between components
[R, P] = corrcoef(prtrain.data)
% The P matrix shows a correlation of R by random chance.
% If p < 0.05 the correlation is significant. 
% Since all values of P are less than 0.05 the values of R are
% significantly correlated. 

