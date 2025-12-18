function h = plot_implicit_surface(phi_fun, alpha, bbox, N, iso, ax, faceAlpha)
% Requires ax. No figure creation. No colorbar/colormap inside.

if nargin < 6 || isempty(ax)
    error('Pass an axes handle: ax = axes; plot_implicit_surface(..., ax)');
end
if nargin < 7 || isempty(faceAlpha), faceAlpha = 0.35; end
if nargin < 5 || isempty(iso), iso = 0; end
if nargin < 4 || isempty(N),   N   = 60; end

% bbox: 3x2 or 1x6
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

xv = linspace(xmin,xmax,N);
yv = linspace(ymin,ymax,N);
zv = linspace(zmin,zmax,N);
[X,Y,Z] = meshgrid(xv,yv,zv);

P = [X(:), Y(:), Z(:)].';     % 3xK (matches your phi_fun convention)
K = size(P,2);
V = zeros(K,1);
for k = 1:K
    V(k) = phi_fun(P(:,k), alpha);
end

%V = phi_fun(P, alpha);
V = reshape(V, size(X));

S = isosurface(X,Y,Z,V,iso);

% force parent = ax (this prevents "random figure" behavior)
h = patch('Parent', ax, 'Faces', S.faces, 'Vertices', S.vertices);
set(h,'EdgeColor','none','FaceAlpha',faceAlpha);

% normals without interp3 grid drama
isonormals(V,h);
end
