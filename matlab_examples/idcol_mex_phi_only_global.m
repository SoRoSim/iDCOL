function phi = idcol_mex_phi_only_global(P, g, alpha, shape_id, params)
%P is 3xN.
%Returns 1xN phi values for the GLOBAL map y = R'*(x-r)/alpha.
%
%Calls: shape_global_ax_mex(g, P, alpha, shape_id, params)

    params  = params(:);
    shape_id = double(shape_id);

    N = size(P,2);
    phi = zeros(1,N);

    try

        out = shape_core_mex('global_xa_phi_grad', g, P, alpha, shape_id, params);

        % Accept common output formats
        if isnumeric(out)
            % could be 1xN, Nx1, or [phi; ...]
            if numel(out) == N
                phi = reshape(out, 1, []);
                return;
            elseif size(out,2) == N
                % if out is 1xN or 4xN etc., take first row as phi
                phi = out(1,:);
                return;
            end
        end

        if isstruct(out) && isfield(out,'phi')
            phi = reshape(out.phi, 1, []);
            return;
        end
    catch
        % fall through to scalar loop
    end

    % Fallback (slow but robust): one point at a time
    for k = 1:N
        outk = shape_core_mex('global_xa_phi_grad', g, P(:,k), alpha, shape_id, params);
        if isnumeric(outk)
            if numel(outk) == 1
                phi(k) = outk;
            else
                phi(k) = outk(1);
            end
        elseif isstruct(outk) && isfield(outk,'phi')
            phi(k) = outk.phi;
        else
            error('idcol_mex_phi_only_global: unsupported mex output format.');
        end
    end
end
