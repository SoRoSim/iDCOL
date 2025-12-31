clc
close all
clear variables

thisFile = mfilename('fullpath');
rootDir  = fileparts(thisFile);

addpath( ...
    fullfile(rootDir,'core'), ...
    fullfile(rootDir,'mex'), ...
    fullfile(rootDir,'examples'), ...
    fullfile(rootDir,'tests') ...
);

addpath(fullfile(rootDir,'eigen-3.4.0'));

fprintf('[Contact-Problem] paths initialized\n');

%build_mex %run if needed