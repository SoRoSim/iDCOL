clc; clear;
% Add project folders (portable)
thisFile = mfilename('fullpath');
testsDir = fileparts(thisFile);          % .../codes/tests
codesDir = fileparts(testsDir);          % .../codes
rootDir  = fileparts(codesDir);          % project root

addpath(fullfile(rootDir,'codes','mex'));   % where your .mexw64 lives

%% ----------------- Polytope 1 in world frame -----------------
A1 = [  1   0    0;   
       -1   0    0;   
        0   1    0;    
        0  -1    0;   
        0   0    1;   
        0   0   -1];

b1 = [1.2; 0.2; 1; 0.2; 1; 0.2];

%% ----------------- Polytope 2 in its local frame ------------

A2 = [  1   0    0;   
       -1   0    0;   
        0   1    0;    
        0  -1    0;   
        0   0    1;   
        0   0   -1];

b2 = [1; 0.2; 1.6; 0.2; 1; 0.2];

%% ----------------- Poses g1, g2 ------------------------------

g1 = eye(4);

Rz = rotz(75);    % degrees
Rx = rotx(30);
R  = Rz * Rx;
r  = [2.2; 0; 0];

g2      = eye(4);
g2(1:3,1:3) = R;
g2(1:3,4)   = r;

%% ----------------- Polytope params for shape_id = 2 ---------

beta = 20;    % smooth-max sharpness

m1 = size(A1,1);
% NOTE: reshape(A1',[],1) so rows go in row-major order (what C++ expects)
params1 = [m1; beta; reshape(A1,[],1); b1];

m2 = size(A2,1);
params2 = [m2; beta; reshape(A2,[],1); b2];

shape_id1 = 2;   % convex polytope with smooth-max (your convention)
shape_id2 = 2;

%% ----------------- Initial guess -----------------------------

x0      = r/2;   % somewhere between the two boxes
alpha0  = 1.0;         % any positive number
lambda10 = 1.0;
lambda20 = 1.0;
lambda0 = 1.0;

% factor = 0.9;
% 
% x0 = [1.184358640482085   0.931875392964412   0.677769445040770]'*factor;
% alpha0   = 0.999505686891195*factor;
% lambda10 = 0.613639860430377*factor;
% lambda20 = 0.682939202470445*factor;

max_iters = 30;
tol       = 1e-10;

%% ----------------- Call Newton-KKT solver --------------------
tic
for i=1:1000
[z_opt, F_opt, J_opt] = idcol_newton_mex( ...
    g1, g2, ...
    shape_id1, params1, ...
    shape_id2, params2, ...
    x0, alpha0, ...
    lambda10, lambda20, max_iters, tol);
end
toc

x_star     = z_opt(1:3);
alpha_star = z_opt(4);
lambda1    = z_opt(5);
lambda2    = z_opt(6);

fprintf('x*      = [%g, %g, %g]^T\n', x_star);
fprintf('alpha*  = %.12g\n', alpha_star);
fprintf('lambda1 = %.6g,  lambda2 = %.6g\n', lambda1, lambda2);
fprintf('||F_opt|| = %.3e\n', norm(F_opt));
%%

[z_opt, F_opt, J_opt] = idcol_newton_s_mex( ...
    g1, g2, ...
    shape_id1, params1, ...
    shape_id2, params2, ...
    x0, alpha0, ...
    lambda10, lambda20, max_iters, tol);


x_star     = z_opt(1:3);
alpha_star = z_opt(4);
lambda1    = z_opt(5);
lambda2    = z_opt(6);

fprintf('x*      = [%g, %g, %g]^T\n', x_star);
fprintf('alpha*  = %.12g\n', alpha_star);
fprintf('lambda1 = %.6g,  lambda2 = %.6g\n', lambda1, lambda2);
fprintf('||F_opt|| = %.3e\n', norm(F_opt));

%%
factor = 0.8; %need good intial guess
x0 = x_star*factor;
alpha0 = alpha_star*factor;
lambda10 = lambda1*factor;
lambda20 = lambda2*factor;


%% ----------------- Sanity check: phi1, phi2 at solution -----

[phi1, g1_vec, ~] = shape_global_ax_mex(g1, x_star, alpha_star, shape_id1, params1);
[phi2, g2_vec, ~] = shape_global_ax_mex(g2, x_star, alpha_star, shape_id2, params2);

fprintf('phi1 = %.3e,  phi2 = %.3e\n', phi1, phi2);

% Check normal equilibrium lambda1*grad_x phi1 + lambda2*grad_x phi2
g1x = g1_vec(1:3);
g2x = g2_vec(1:3);
eq_norm = norm(lambda1*g1x + lambda2*g2x);
fprintf('||lambda1*grad_x phi1 + lambda2*grad_x phi2|| = %.3e\n', eq_norm);
