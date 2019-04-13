clear all
%% /Users/rafiul_rasel/Desktop/Machine_Learning/LinearRegression/MultiVariable


y = [0 0.5 1 2];
len = length(y);
x = [1 1 1 1];
x = [x; [1 2 2.4 3]];
x = [x; [0 1 2 4]];


t = [0, 0, 0];
ts = t;
al = 0.01;

count = 1;

%% cost function J(t0, t1) = 1/2m * sum((t0+t1x - y)^2)
%% th_j = th_j - alpha*1/m sum((th^T*x_j-y).*x_j)

while(1)
ts = t;
for i=1:length(t)
t(i) = t(i) - al*(1/len)*sum((t*x-y).*x(i,:));
end

if(norm(ts-t)<1e-5)
break;
end
end

disp(1);
scatter(x(3,:),y); hold on;
y1 = t*x;

plot(x(3,:), y1); hold off;






