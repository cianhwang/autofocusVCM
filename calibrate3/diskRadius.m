function [r,alpha] = diskRadius(f0_idx, f_curr_idx)
% unit: mm
f = 4.52;
f_number = 2;
D = f/f_number;
delta = 1.55e-3;

% rangeDictInch = {'900': 3.25,
%     '800': 4,
%     '700': 5,
%     '650': 6,
%     '600': 7.75,
%     '550': 12,
%     '525': 14,
%     '500': 22,
%     '495': 27,
%     '490': 33,
%     '480': 40.5,
%     '475': 54.5}
keySet = {'1000', '875', '800','725','675','650',...
    '600','550','525','500','475'};
valset = [0.25, 0.75, 1, 1.625, 2.5, 3.125, 4.25, 8, 11.75, 24, 46.75];
keySet = {'475', '500', '525', '550', '575', '600', '625', '650', '675', '700', '725', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'};
valset = [46.7500000000000	24	11.7500000000000	8	6.12500000000000	4.25000000000000	3.68750000000000	3.12500000000000	2.50000000000000	2.06250000000000	1.62500000000000	1.41666666666667	1.20833333333333	1	0.916666666666667	0.833333333333333	0.750000000000000	0.650000000000000	0.550000000000000	0.450000000000000	0.350000000000000	0.250000000000000];


valSet = (2.75+valset).*25.4; %inch to mm
rangeDict = containers.Map(keySet,valSet);

f0 = rangeDict(f0_idx); %in-focus distance between object and camera
f_curr = rangeDict(f_curr_idx); % object distance
z = 1/(1/f-1/f_curr); % image distance
z0 = 1/(1/f-1/f0); % distance between lens and sensor
D_curr = abs(z0-z)/z*D; % disk blur range
r = ceil(D_curr/delta/2)
alpha = z0/z
end