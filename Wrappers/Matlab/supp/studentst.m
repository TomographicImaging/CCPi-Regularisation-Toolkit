function [f,g,h,s,k] = studentst(r,k,s)
% Students T penalty with 'auto-tuning'
%
% use:
%   [f,g,h,{k,{s}}] = studentst(r)     - automatically fits s and k
%   [f,g,h,{k,{s}}] = studentst(r,k)   - automatically fits s
%   [f,g,h,{k,{s}}] = studentst(r,k,s) - use given s and k
%
% input:
%   r - residual as column vector
%   s - scale (optional)
%   k - degrees of freedom (optional)
% 
% output:
%   f - misfit (scalar)
%   g - gradient (column vector)
%   h - positive approximation of the Hessian (column vector, Hessian is a diagonal matrix)
%   s,k - scale and degrees of freedom
%
% Tristan van Leeuwen, 2012.
% tleeuwen@eos.ubc.ca

% fit both s and k
if nargin == 1
    opts = optimset('maxFunEvals',1e2);
    tmp = fminsearch(@(x)st(r,x(1),x(2)),[1;2],opts);
    s   = tmp(1);
    k   = tmp(2);
end


if nargin == 2
    opts = optimset('maxFunEvals',1e2);
    tmp = fminsearch(@(x)st(r,x,k),[1],opts);
    s   = tmp(1);
end

% evaulate penalty
[f,g,h] = st(r,s,k);


function [f,g,h] = st(r,s,k)
n = length(r);
c = -n*(gammaln((k+1)/2) - gammaln(k/2) - .5*log(pi*s*k));
f = c + .5*(k+1)*sum(log(1 + conj(r).*r/(s*k)));
g = (k+1)*r./(s*k + conj(r).*r);
h = (k+1)./(s*k + conj(r).*r);
