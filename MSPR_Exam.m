clc, clear all, close all;

% Removing prwaitbar to reduce computation time
% prwaitbar off 

% Surpress warnings
prwarning(0);

%open file and load in data to Idat
fid = fopen('Skin_NonSkin.txt');
Idat = textscan(fid, '%f%f%f%f','delimiter',' ');
fclose(fid);

%Seperate data from classifier.
data=[Idat{1} Idat{2} Idat{3}];

% Generate prdata
prdat = prdataset(data, Idat{4}, 'featlab',['red  ';'green';'blue '], 'lablist',['skin   ';'nonskin']);

% Generate training set and data set with a 80-20% ratio.
trainSize = round(length(data)*0.8);
[prtrain, prtest]=gendat(prdat,trainSize);

% Get a smaller sample of data from the prtest set. For some calculations
% the test set is too big and MATLAB runs out of memory. 
dataSmall = datasample([data Idat{4}],1000);

% Generate small set of prdata. 
prdatSmall = prdataset([dataSmall(:,1) dataSmall(:,2) dataSmall(:,3)], dataSmall(:,4), 'featlab',['red  ';'green';'blue '], 'lablist',['skin   ';'nonskin']);

%% Plot components in a scatter plot against each other. 
% Computation time: 7.130323 seconds
colors = {'red' 'green' 'blue'};
figure;
for i=2:3,
    for j=1:i-1,
        if(j==1&&i==3)
            subplot(3,1,3)
        else
            subplot(3,1,j)
        end
        scatterd(prtrain(:,[i j]),'legend')
        title(sprintf('Feature %s vs. %s',colors{i},colors{j}))
        % reverse the plot order. Plot skin on top on non-skin. 
        set(gca,'Children',flipud(get(gca,'Children'))); 
    end
end

% Skin (blue +) is more grouped together in all three scenarios then
% nonskin (red *). Tentative analysis would suggest that blue (+) is
% skin and the red (*) is nonskin. The red (*) data is more spread out of
% some clusters seem random whereas blue (+) is grouped. 

%% Correlation between components
[R, P] = corrcoef(prtrain.data);
% The P matrix shows a correlation of R by random chance.
% If p < 0.05 the correlation is significant. 
% Since all values of P are less than 0.05 the values of R are
% significantly correlated. 



%% Principal Component Analysis (p. 216)

classes = {'prtrain', 'prdatSmall'};
for i=1:length(classes)
    if strcmp(classes{i},'prtrain')
        [v,frac] = pcam(prtrain,0);
    elseif strcmp(classes{i},'prdatSmall')
        [v,frac] = pcam(prdatSmall,0);
    end
    figure
    plot(v)
    title(['Cumulative Eigenvalues (' classes{i} ')'])
    ylabel('Percentage of Explained Variance')
    xlabel('No of Principle Components')
end

figure
for i=1:length(classes)
% Plot the PCA scaled to the mean
    if strcmp(classes{i},'prtrain')
        w= pcam(prtrain,2);
        XR = prtrain*w;
    elseif strcmp(classes{i},'prdatSmall')
        w= pcam(prdatSmall,2);
        XR = prdatSmall*w;
    end
   
    subplot(2,1,i)
    scatterd(XR,'legend')
% Reverse the plot order. Plot skin on top on non-skin. 
    set(gca,'Children',flipud(get(gca,'Children'))); 
    
% Plot the unit vectors of the components overlayed the PCA
    hold on;
    factor = 50;
    for j=1:3,
        line([0 w.data.rot(j,1)*factor],[0 w.data.rot(j,2)*factor],'lineWidth',2,'color','g')
        text(w.data.rot(j,1)*factor,w.data.rot(j,2)*factor,['   '  w.data.lablist_in(j,:)])
    end
    scatter(w.data.rot(:,1)*factor,w.data.rot(:,2)*factor,'*k')
    xlabel('Component 1')
    ylabel('Component 2')
    title(['Principal Component Analysis Plot (' classes{i} ')']);
end

% Looking at the skin (blue +) it is possible to see based on the unit
% vectors that the blue color channel is dominent, whereas the red color
% channel is much less present. For the nonskin (red *) this is opposite, 
% there we have green and red and the dominent color channel and blue as 
% less present one.

%% PCA plot
varLabels={'Red','Green','Blue'};
classes = {'prtrain', 'prdatSmall'};
figure;
for i=1:length(classes)
    subplot(2,1,i);
    if strcmp(classes{i},'prtrain')
        [R, P] = corrcoef(prtrain.data);
    elseif strcmp(classes{i},'prdatSmall')
        [R, P] = corrcoef(prdatSmall.data);
    end
    R = [R R(:,end); R(end,1:end) R(end,end)]; %add one row bcus pcolor removes 1
    pcolor(R); axis xy; grid on; colorbar;
    % create place for labels showing dimension
    set(gca,'YTick',[1:length(varLabels)]+0.5,'YTickLabel',varLabels,'FontSize',12);
    set(gca,'XTick',[1:length(varLabels)]+0.5,'XTickLabel',varLabels,'FontSize',12);
    title(['Correlation Coefficients Plot (' classes{i} ')'],'fontSize',20);
end
%% Feature Selection (201-202)
wld=ldc([]);
[wfs R]=featselm(prtrain,wld,'forward',2);
figure;
scatterd(prtrain*wfs,'legend');
title(['forward: ' num2str(+wfs{1})]);
set(gca,'Children',flipud(get(gca,'Children'))); 

% Using featselm we extract the two important features from the dataset
% using the wrapper method for linear distriminant analysis and the forward
% search. The two selected features are Red and Blue. If plottet without using
% Feature Selection Red and Green would be selected. 


%% Clustering (226)

for i=2:15
    tic
    [idx]=prkmeans(prdatSmall,i);
    figure;
    scatter(prdatSmall(:,1),prdatSmall(:,3),[],idx);
    title([num2str(i) ' Groups'])
    toc
end
% Plotting the training set with clusters from 2 to 10. 
% Since the Feature Selection said that feature 1 (red) and feature 3
% (blue) were the most important ones, we plot them vs each other and try
% and determine 10 different clusters. 

%% Gap Statistics
% Computation time: 193.713183 seconds
tic
eva = evalclusters(prdatSmall.data,'kmeans','gap','KList',[1:15])
figure;
plot(eva)
toc
% Evalclusters takes the training set and automatically finds out how many
% clusters are optimal from 1 to 10. This unfortunately crashes matlab
% because it runs out of memory (takes up >8GB!). So somehow we need to do
% some manual clustering. 

% Edit: Since we have created a smaller dataset to work with (prdatSmall)
% we can use the evalclusters on that set. Using up to 15 clusters we try
% and see what the optimal number of clusers would be. 

% Result shows that 13 clusters would be optimal on a number from 1 to 15.

%% Classification 
% Computation time: 160.6440 seconds.

% The big part is classification, we are going to use both parametric (ldc, qdc, nmc) and
% non-parametric (dtc, perlc, parzenc, knnc) classifiers. Some classifiers cannot
% run with the original dataset (prdat), so they will run with the scaled
% down dataset (prdatSmall). 

figure;
class_modes={'ldc', 'qdc', 'dtc', 'perlc', 'fisherc', 'udc', 'nmc', 'parzenc', 'knnc'};
for i=1:length(class_modes)
    A=prdat(:,[1 3]);
    subplot(3,3,i);
    if (strcmp(class_modes{i},'parzenc') ~=1 || strcmp(class_modes{i},'perlc') ~=1 || strcmp(class_modes{i},'knnc') ~=1)
        scatterd(A,'legend');
    end
    if strcmp(class_modes{i},'ldc')
        plotc(A*ldc);
        [e,ce,nlab_out]= prcrossval(prtrain,ldc([]),4);
        ac(i)=1-e;
        title(['Linear Bayes Normal (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'qdc')
        plotc(A*qdc);
        [e,ce,nlab_out]= prcrossval(prtrain,qdc([]),4);
        ac(i)=1-e;
        title(['Quadratic Bayes Normal (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'dtc')
        plotc(A*dtc);
        [e,ce,nlab_out]= prcrossval(prtrain,dtc([]),4);
        ac(i)=1-e;
        title(['Decision Tree (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'perlc')
        A=prdatSmall(:,[1 3]);
        scatterd(A,'legend');
        plotc(perlc(A,10,0.01));
        [e,ce,nlab_out]= prcrossval(prdatSmall,perlc([]),4);
        ac(i)=1-e;
        title(['Linear Perceptron (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'fisherc')
        plotc(fisherc(A));
        [e,ce,nlab_out]= prcrossval(prtrain,fisherc([]),4);
        ac(i)=1-e;
        title(['Linear Discriminant (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'udc')
        plotc(udc(A));
        [e,ce,nlab_out]= prcrossval(prtrain,udc,4);
        ac(i)=1-e;
        title(['Uncorrelated normal based quadratic Bayes (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'nmc')
        plotc(nmc(A));
        [e,ce,nlab_out]= prcrossval(prtrain,nmc([]),4);
        ac(i)=1-e;
        title(['Nearest Mean (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'parzenc')
        A=prdatSmall(:,[1 3]);
        scatterd(A,'legend');
        plotc(parzenc(A,1));
        [e,ce,nlab_out]= prcrossval(prdatSmall,parzenc,4);
        ac(i)=1-e;
        title(['Parzen (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    elseif strcmp(class_modes{i},'knnc')
        A=prdatSmall(:,[1 3]);
        scatterd(A,'legend');
        plotc(knnc(A));
        [e,ce,nlab_out]= prcrossval(prdatSmall,knnc,4);
        ac(i)=1-e;
        title(['K-Nearest Neighbor (' class_modes{i} '), Ac: ' num2str(ac(i))]);
    end
    if (strcmp(class_modes{i},'parzenc') || strcmp(class_modes{i},'perlc') || strcmp(class_modes{i},'knnc'))
        C{i}=confmat(prdatSmall.nlab,nlab_out);
    else 
        C{i}=confmat(prtrain.nlab,nlab_out);
    end
end 