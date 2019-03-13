clc; clear; close all;

keySet = {'1000', '875', '800','725','675','650',...
    '600','550','525','500','475'};
figure;
y = 0;
for i = 1: 10
   load(strcat('Model_parameters_', keySet{i}, '.mat'));
   scatter(Model_parameters(:, 2), (Model_parameters(:, 4)));
   y = y + Model_parameters(:, 4);
   hold on;
end
x = Model_parameters(:, 2);
y = y*10-100;