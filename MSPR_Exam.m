clear all;
prwaitbar off
fid = fopen('Skin_NonSkin.txt');
Idat = textscan(fid, '%f%f%f%s','delimiter',' ');
fclose(fid);

data=[Idat{1} Idat{2} Idat{3}];
prdat=prdataset(data, Idat{4}, 'featlab',['r';'g';'b'], 'lablist',['skin   ';'nonskin'])