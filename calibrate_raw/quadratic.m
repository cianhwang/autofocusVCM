clc; clear; close all;

keySet = {'1000','950', '875', '825','800','725','700', '675','650',...
    '625', '600','575', '550','525','500','475'};
figure;
y = 0;
for i = 1: length(keySet)
   load(strcat('Model_parameters_B_', keySet{i}, '.mat'));
   scatter(Model_parameters_B(:, 4), (Model_parameters_B(:, 3)), [], 'B');
%    y = y + Model_parameters(:, 4);
   hold on;
  load(strcat('Model_parameters_G_', keySet{i}, '.mat'));
   scatter(Model_parameters_G(:, 4), (Model_parameters_G(:, 3)), [], 'G');
%    y = y + Model_parameters(:, 4);
   hold on;
  load(strcat('Model_parameters_R_', keySet{i}, '.mat'));
   scatter(Model_parameters_R(:, 4), (Model_parameters_R(:, 3)), [], 'R');
%    y = y + Model_parameters(:, 4);
   hold on;
end
% x = Model_parameters(:, 2);
% y = y/10-1;   