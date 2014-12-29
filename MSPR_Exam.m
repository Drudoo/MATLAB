clear all;
prwaitbar off
fid = fopen('Skin_NonSkin.txt');
Idat = textscan(fid, '%f%f%f%f','delimiter',' ');
fclose(fid);