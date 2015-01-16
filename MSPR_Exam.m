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

%% Principal Component Analysis (p. 216)
[v,frac] = pcam(prtrain,0);
figure
plot(v)
title('Cumulative Eigenvalues')
ylabel('Percentage of Explained Variance')
xlabel('No of Principle Components')

%Plot the PCA scaled to the mean 
w= pcam(prtrain,3);
XR = prtrain*w;
figure
scatterd(XR,'legend')

%labels = num2str((1:size(XR,1))','%d');    %'
%text(XR.data(:,1), XR.data(:,2), labels, 'horizontal','left', 'vertical','bottom')

%Plot the unit vectors of the components overlayed the PCA
hold on;
factor = 50;
for i=1:3,
    line([0 w.data.rot(i,1)*factor],[0 w.data.rot(i,2)*factor],'lineWidth',2,'color','g')
    text(w.data.rot(i,1)*factor,w.data.rot(i,2)*factor,['   '  w.data.lablist_in(i,:)])
end
scatter(w.data.rot(:,1)*factor,w.data.rot(:,2)*factor,'*k')
xlabel('Component 1')
ylabel('Component 2')

% Looking at the skin (blue +) it is possible to see based on the unit
% vectors that the blue color channel is dominent, whereas the red color
% channel is much less present. For the nonskin (red *) this is opposite, 
% there we have green and red and the dominent color channel and blue as 
% less present one.

%% Feature Selection (201-202)
wld=ldc([]);
wfs=featselm(prtrain,wld,'forward',2)
figure;
scatterd(prtrain*wfs);
title(['forward: ' num2str(+wfs{1})]);

% Using featselm we extract the two important features from the dataset
% using the wrapper method for linear distriminant analysis and the forward
% search. The two selected features are Red and Blue. These two features
% also have the largest standard deviation. If plottet without using
% Feature Selection Red and Green would be selected. 


%%
