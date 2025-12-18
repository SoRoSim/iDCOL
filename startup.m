thisFile = mfilename('fullpath');
rootDir  = fileparts(thisFile);

addpath( ...
    fullfile(rootDir,'core'), ...
    fullfile(rootDir,'mex'), ...
    fullfile(rootDir,'examples'), ...
    fullfile(rootDir,'tests') ...
);

% Eigen is header-only; path needed only for MEX builds,
% but adding it does no harm and avoids confusion.
addpath(fullfile(rootDir,'eigen-3.4.0'));

fprintf('[Contact-Problem] paths initialized\n');

%% To compile MEX files

% mex -O -outdir mex -I"." -I".\eigen-3.4.0" mex\idcol_newton_mex.cpp core\shape_core.cpp core\idcol_kkt.cpp core\idcol_newton.cpp
% mex -O -outdir mex -I"." -I".\eigen-3.4.0" mex\idcol_KKT_FJ_mex.cpp core\shape_core.cpp core\idcol_kkt.cpp
% mex -O -outdir mex -I"." -I".\eigen-3.4.0" mex\shape_global_ax_mex.cpp core\shape_core.cpp
% mex -O -outdir mex -I"." -I".\eigen-3.4.0" mex\shape_local_mex.cpp core\shape_core.cpp