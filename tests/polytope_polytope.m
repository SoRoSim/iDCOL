%clc; %clear;

%% ----------------- Polytope 1 in world frame -----------------
A1 = [  1   0    0;   
       -1   0    0;   
        0   1    0;    
        0  -1    0;   
        0   0    1;   
        0   0   -1];

b1 = [1; 1; 1; 1; 1; 1];

%% ----------------- Polytope 2 in its local frame ------------

A2 = [  1   0    0;   
       -1   0    0;   
        0   1    0;    
        0  -1    0;   
        0   0    1;   
        0   0   -1];

b2 = [1; 1; 2; 2; 1.5; 1.5];

%% ----------------- Poses g1, g2 ------------------------------

% g1 = eye(4);
% 
% Rz = rotz(rand*360);    % degrees
% Rx = rotx(rand*360);
% Ry = roty(rand*360);
% R  = Rx * Ry * Rz;
% r  = [1+2*rand; 1+2*rand; 1+2*rand];
% 
% g2      = eye(4);
% g2(1:3,1:3) = R;
% g2(1:3,4)   = r;

g1 = eye(4);

% --- translation: uniform on a spherical shell around body 1 ---
rmin = 0.1;      % choose
rmax = 3.0;      % choose

u = randn(3,1);          % isotropic direction
u = u / norm(u);

rho = (rmax^3 + (rmin^3 - rmax^3)*rand)^(1/3);   % uniform in volume
% If you want uniform in radius instead, use: rho = rmin + (rmax-rmin)*rand;

r = rho * u;

% --- orientation: uniform random rotation (SO(3)) via random quaternion ---
q = randn(4,1);
q = q / norm(q);
w=q(1); x=q(2); y=q(3); z=q(4);

R = [1-2*(y^2+z^2)   2*(x*y - z*w)   2*(x*z + y*w);
     2*(x*y + z*w)   1-2*(x^2+z^2)   2*(y*z - x*w);
     2*(x*z - y*w)   2*(y*z + x*w)   1-2*(x^2+y^2)];

g2 = eye(4);
g2(1:3,1:3) = R;
g2(1:3,4)   = r;


%% ----------------- Polytope params for shape_id = 2 ---------

beta = 20;    % smooth-max sharpness

m1 = size(A1,1);
% NOTE: reshape(A1',[],1) so rows go in row-major order
params1 = [m1; beta; reshape(A1,[],1); b1]; %column major

m2 = size(A2,1);
params2 = [m2; beta; reshape(A2,[],1); b2];

shape_id1 = 2;   % convex polytope with smooth-max (your convention)
shape_id2 = 2;

%% ----------------- Initial guess -----------------------------

x0      = r/2;   % somewhere between the two boxes
alpha0  = 1.0;         % any positive number
lambda10 = 1.0;
lambda20 = 1.0;

%%
L = 1; %Length scale, replace with L = max(R1,R2); %bounding sphere radii
max_iters = 30;
tol       = 1e-10;

%%

% tic
% for i=1:1000
[z_opt, F_opt, J_opt] = idcol_newton_mex( ...
    g1, g2, ...
    shape_id1, params1, ...
    shape_id2, params2, ...
    x0, alpha0, ...
    lambda10, lambda20, L, max_iters, tol);
% end
% toc

x_star     = z_opt(1:3);
alpha_star = z_opt(4);
lambda1    = z_opt(5);
lambda2    = z_opt(6);

% fprintf('x*      = [%g, %g, %g]^T\n', x_star);
% fprintf('alpha*  = %.12g\n', alpha_star);
% fprintf('lambda1 = %.6g,  lambda2 = %.6g\n', lambda1, lambda2);
% 
% fprintf('||F_opt|| = %.3e\n', norm(F_opt));


%% FSOLVE

z0 = [x0(:); alpha0; lambda10; lambda20];

fun = @(z) idcol_kkt_FJ_mex(z, g1, g2, shape_id1, shape_id2, params1, params2);

opts = optimoptions('fsolve', ...
    'SpecifyObjectiveGradient', true, ...
    'Display', 'none', ...
    'FunctionTolerance', 1e-12, ...
    'StepTolerance', 1e-12, ...
    'OptimalityTolerance', 1e-12, ...
    'MaxIterations', 200, ...
    'MaxFunctionEvaluations', 2000);
tic
for i=1:1
[zsol, Fval, exitflag, output] = fsolve(fun, z0, opts);
end
%toc
% exitflag
% zsol
%% ----------------- Sanity check: phi1, phi2 at solution -----

[phi1, g1_vec, ~] = shape_global_ax_mex(g1, x_star, alpha_star, shape_id1, params1);
[phi2, g2_vec, ~] = shape_global_ax_mex(g2, x_star, alpha_star, shape_id2, params2);

% fprintf('phi1 = %.3e,  phi2 = %.3e\n', phi1, phi2);

% Check normal equilibrium lambda1*grad_x phi1 + lambda2*grad_x phi2
g1x = g1_vec(1:3);
g2x = g2_vec(1:3);
eq_norm = norm(lambda1*g1x + lambda2*g2x);
% fprintf('||lambda1*grad_x phi1 + lambda2*grad_x phi2|| = %.3e\n', eq_norm);

%% Plotting

plotit = true;


if plotit

% smax = @(z) (1/beta) * log(sum(exp(beta*z)));  % z is a vector
smax = @(z) max(z) + (1/beta) * log(sum(exp(beta*(z-max(z)))));  % z is a vector

% phi_1(x/alpha) and phi_2(R'(x-r)/alpha)

phi1 = @(x,alpha) smax(A1*(x/alpha)              - b1);
phi2 = @(x,alpha) smax(A2*(R'*(x - r)/alpha)     - b2);


figure(1); clf;
ax = axes('Parent', gcf); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal'); view(ax,3);

xyzlim = 5;

h1 = plot_implicit_surface(phi1, 1.0, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 80, 0, ax);
set(h1,'FaceColor',[0 0.6 1]);

h2 = plot_implicit_surface(phi2, 1.0, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 80, 0, ax);
set(h2,'FaceColor',[1 0.5 0]);

%scaled
h3 = plot_implicit_surface(phi1, alpha_star, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 80, 0, ax);
set(h1,'FaceColor','r');

h4 = plot_implicit_surface(phi2, alpha_star, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 80, 0, ax);
set(h2,'FaceColor','g');

%kissing point
plot3(x_star(1), x_star(2), x_star(3), 'r.', 'MarkerSize', 25);

% Axis labels (LaTeX)
xlabel(ax,'$x\;(\mathrm{m})$','Interpreter','latex');
ylabel(ax,'$y\;(\mathrm{m})$','Interpreter','latex');
zlabel(ax,'$z\;(\mathrm{m})$','Interpreter','latex');



% Title based on alpha_star (LaTeX)
tol = 1e-6;
if alpha_star > 1 + tol
    contactStr = '(Separated)';
elseif abs(alpha_star - 1) <= tol
    contactStr = '(Contact)';
else
    contactStr = '(Penetration)';
end

tstr = sprintf('$\\alpha^* = %.4f\\quad %s$', alpha_star, contactStr);
title(ax, tstr, 'Interpreter','latex');


axis tight
camlight(ax,'headlight'); lighting(ax,'gouraud');


rout= 2.60443; [X,Y,Z]=sphere(40); surf(rout*X+r(1),rout*Y+r(2),rout*Z+r(3),'FaceAlpha',0.3,'EdgeColor','none');

end