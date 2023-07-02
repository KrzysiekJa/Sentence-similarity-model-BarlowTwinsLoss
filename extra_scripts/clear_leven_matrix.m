clearvars; close all; clc;

img = imread('img.png');

img(img>210 & img<250) = 256;

imshow(img);
