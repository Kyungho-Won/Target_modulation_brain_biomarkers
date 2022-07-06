function CSP_out = extract_CSP(x1_in, x2_in)
% CSP_out = extract_CSP(x1_in, x2_in)
% input:
%   x1_in: 3D data for class1 [ch, time, trial]
%   x2_in: 3D data for class2 [ch, time, trial]
%  ** This function assumes x1_in and x2_in dimensions are identical.
%
% output:
%   CSP_out.W: CSP filters [ch, ch]
%   CSP_out.D: eigen values [ch, 1]

% return error if dimensions are different
if size(x1_in) ~= size(x2_in)
    error('Error: Input data have different dimension.');
end
nCh = size(x1_in, 1);
nTime = size(x1_in,2);
nTrial_x1 = size(x1_in, 3);
nTrial_x2 = size(x2_in, 3);

% step1. concatenate training data -> [ch, (timex trial)]
concat_x1 = reshape(x1_in, [nCh, nTime*nTrial_x1]); 
concat_x2 = reshape(x2_in, [nCh, nTime*nTrial_x2]);

% step2. extract CSP
cov_x1 = concat_x1 * concat_x1'; 
cov_x2 = concat_x2 * concat_x2';

cov_x1x2 = cov_x1+cov_x2;
[W, D] = eig(cov_x1, cov_x1x2);

CSP_out.W = W;
CSP_out.D = D;
end