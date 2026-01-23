function [h, V, F] = plot_implicit_surface(phi_fun, alpha, bbox, N, iso, ax, faceAlpha)
%PLOT_IMPLICIT_SURFACE  Plot implicit iso-surface phi(x,alpha)=iso.
% Vectorized evaluation (no per-point loop). Requires ax.
%
% Inputs:
%   phi_fun   : function handle, supports phi_fun(P, alpha) where P is 3xK
%   alpha     : scalar
%   bbox      : 3x2 or 1x6
%   N         : grid resolution per axis (default 60)
%   iso       : iso level (default 0)
%   ax        : axes handle (REQUIRED)
%   faceAlpha : patch transparency (default 0.35)
%
% Outputs:
%   h : patch handle
%   V : vertices (Nv x 3)
%   F : faces (Nf x 3)

    if nargin < 6 || isempty(ax)
        error('Pass an axes handle: ax = axes; plot_implicit_surface(..., ax)');
    end
    if nargin < 7 || isempty(faceAlpha), faceAlpha = 0.35; end
    if nargin < 5 || isempty(iso), iso = 0; end
    if nargin < 4 || isempty(N),   N   = 60; end

    % ---- bbox: 3x2 or 1x6 ----
    if isequal(size(bbox),[3 2])
        xmin=bbox(1,1); xmax=bbox(1,2);
        ymin=bbox(2,1); ymax=bbox(2,2);
        zmin=bbox(3,1); zmax=bbox(3,2);
    elseif isvector(bbox) && numel(bbox)==6
        bbox=bbox(:).';
        xmin=bbox(1); xmax=bbox(2);
        ymin=bbox(3); ymax=bbox(4);
        zmin=bbox(5); zmax=bbox(6);
    else
        error('bbox must be 3x2 or 1x6.');
    end

    % ---- grid ----
    xv = linspace(xmin,xmax,N);
    yv = linspace(ymin,ymax,N);
    zv = linspace(zmin,zmax,N);
    [X,Y,Z] = meshgrid(xv,yv,zv);

    P = [X(:)'; Y(:)'; Z(:)'];   % 3xK (matches your convention)

    % ---- vectorized phi eval (correct syntax) ----
    Vphi = phi_fun(P, alpha);    % should return 1xK or Kx1
    Vphi = Vphi(:);
    Vgrid = reshape(Vphi, size(X));

    % ---- extract iso-surface ----
    S = isosurface(X,Y,Z,Vgrid,iso);

    if isempty(S.vertices) || isempty(S.faces)
        error('plot_implicit_surface: empty isosurface. Increase bbox or N, or check iso.');
    end

    V = S.vertices;
    F = S.faces;

    % ---- plot into provided axes ----
    h = patch('Parent', ax, 'Faces', F, 'Vertices', V);
    set(h,'EdgeColor','none','FaceAlpha',faceAlpha);

    % normals (avoid interp3 issues by using grid volume)
    isonormals(X,Y,Z,Vgrid,h);
end
