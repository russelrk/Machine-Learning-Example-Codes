clear all
%% /Users/rafiul_rasel/Desktop/Machine_Learning/LinearRegression/MultiVariable


y = [0; 0.5; 1; 1.9; 2.5; 3; 3.4; 3.9; 4.2; 4.5];
len = length(y);
x = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1;];
x = [x, [1; 2; 2.4; 3; 4; 4.5; 5.3; 6; 6.7; 7.5]];
x = [x, [0; 1; 2; 4; 5; 6; 7; 8; 9; 10]];

%% least square norm method
%% x.'*x can be singular, use tikhonov regularizaiton to regularize 
t = pinv(x.'*x)*x.'*y;

disp(1);
scatter(x(:,2),y); hold on;
y1 = x*t;

plot(x(:,2), y1); hold off;






