function build_mex()
    thisFile = mfilename('fullpath');
    rootDir  = fileparts(thisFile);

    inc1  = ['-I' rootDir];
    inc2  = ['-I' fullfile(rootDir,'eigen-3.4.0')];
    cxx17 = 'COMPFLAGS=$COMPFLAGS /std:c++17';

    outd = fullfile(rootDir,'mex');
    if ~exist(outd,'dir'), mkdir(outd); end

    mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
        'mex\idcol_solve_mex.cpp','core\shape_core.cpp','core\idcol_kkt.cpp','core\idcol_newton.cpp');

    % mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
    %     'mex\idcol_kkt_mex.cpp','core\shape_core.cpp','core\idcol_kkt.cpp');
    % 
    % mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
    %     'mex\shape_core_mex.cpp','core\shape_core.cpp');
    % 
    % mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
    %     'mex\radial_bounds_mex.cpp','core\shape_core.cpp','core\radial_bounds.cpp');

    fprintf('[Contact-Problem] MEX build complete\n');
end
