function g = filtersinc(PR)


% filtersinc.m
%
% Written by Waqas Akram
%
% "a":	This parameter varies the filter magnitude response.
%	When "a" is very small (a<<1), the response approximates |w|
%	As "a" is increased, the filter response starts to 
%	roll off at high frequencies.
a = 1;

[Length, Count] = size(PR);
w = [-pi:(2*pi)/Length:pi-(2*pi)/Length];

rn1 = abs(2/a*sin(a.*w./2));
rn2 = sin(a.*w./2);
rd = (a*w)./2;
r = rn1*(rn2/rd)^2;

f = fftshift(r);
for i = 1:Count
        IMG = fft(PR(:,i));
        fimg = IMG.*f';
        g(:,i) = ifft(fimg);
end
g = real(g);