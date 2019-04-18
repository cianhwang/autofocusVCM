clc; clear; close all;

keySet = {'1000','950', '875', '825','800','725','700', '675','650',...
    '625', '600','575', '550','525','500','475'};

%% alpha-s
% figure;
y = 0;
xx = []
yy = []
for i = 1:16
    
   load(strcat('Model_parameters_BGR_', keySet{i}, '.mat'));
%    scatter(Model_parameters_BGR(:, 2), (Model_parameters_BGR(:, 4)));
   xx = [xx; Model_parameters_BGR(:, 2)];
   yy = [yy; Model_parameters_BGR(:, 4)-min(Model_parameters_BGR(:, 4))];
   y = y + Model_parameters_BGR(:, 4);
%    hold on;
 
end
figure, 
scatter(xx,yy)
hold on
[p,S] = polyfit(xx, yy, 2);
x1 = linspace(475, 1000);
[y1, delta] = polyval(p,x1,S);
plot(x1,y1+2*delta,'m--',x1,y1-2*delta,'m--')
axis([475, 1000, 0, 0.1])

% figure;
% x = Model_parameters_BGR(:, 2);
% y = y / 16;
% scatter(x, y);
% hold on
% [p,S] = polyfit(x, y, 2);
% x1 = linspace(475, 1000);
% [y1, delta] = polyval(p,x1,S);
% plot(x1, y1, x1(37), y1(37), '*');
% plot(x1,y1+2*delta,'m--',x1,y1-2*delta,'m--')
% legend('\alpha-s','quadratic fitting', '(665.9, 1.0)', '95% prediction interval');
% xlabel('Steps');
% ylabel('\alpha');
% title('\alpha - s Relationship')

%% alpha-r
%  figure,
%  x = [];
%  y = [];
% for i = 2:15
%    load(strcat('Model_parameters_BGR_', keySet{i}, '.mat'));
%    x = [x;abs(Model_parameters_BGR(2:end-1, 4)-1)];
%    y = [y;Model_parameters_BGR(2:end-1, 3)];
%    %axis([0 0.1 0 50])
% end
% scatter(y, x, [], 'b');
% hold on;
% [p,S] = polyfit(y, x, 1);
% y1 = linspace(0, 50);
% [x1, delta] = polyval(p,y1,S);
% plot(y1,x1+2*delta,'m--',y1,x1-2*delta,'m--')
% axis([0 50 0 0.1])
% title('Blue channel (focal step: 500-950)')
% ylabel('1-\alpha')
% xlabel('r')
% legend(strcat('delta =',num2str(delta(1),3)));
% 
% figure,
% scatter(x, y, [], 'b');
% hold on;
% [p,S] = polyfit(x, y, 1);
% x1 = linspace(0, 50);
% [y1, delta] = polyval(p,x1,S);
% plot(x1,y1+2*delta,'m--',x1,y1-2*delta,'m--')
% axis([0 0.1 0 50])
% title('Blue channel (focal step: 500-950)')
% xlabel('1-\alpha')
% ylabel('r')
% legend(strcat('delta =',num2str(delta(1))));

%% visualize 0.01
% I = imread("5_800_10093_s01_00000.jpg");
% [m n ~] = size(I);
% I = I(m/2-256-111:m/2+255-111, n/2-256-111:n/2+255-111);
% figure, imshow(I);
% h = fspecial('disk',3);
% I2 = imfilter(I, h);
% figure, imshow(I2)
% I3 = imfilter(I2, h);
% h2 = fspecial('disk',4);
% I4 = imfilter(I, h2);
% montage({I3, I4})