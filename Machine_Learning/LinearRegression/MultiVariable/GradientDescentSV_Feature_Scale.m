clear all
%% /Users/rafiul_rasel/Desktop/Machine_Learning/LinearRegression/MultiVariable


y = [0; 0.5; 1; 1.9; 2.5; 3; 3.4; 3.9; 4.2; 4.5];
len = length(y);
x = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1;];
x = [x, [1; 2; 2.4; 3; 4; 4.5; 5.3; 6; 6.7; 7.5]];
x = [x, [0; 1; 2; 4; 5; 6; 7; 8; 9; 10]];


%% feature
for i=2:length(x(1,:))
x(:,i) = (x(:,i) - mean(x(:,i)))/(max(x(:,i))-min(x(:,i)));
end



t = zeros(1,length(x(1,:))).';
ts = t;
al = 1;

count = 1;

%% cost function J(t0, t1) = 1/2m * sum((t0+t1x - y)^2)
%% th_j = th_j - alpha*1/m sum((th^T*x_j-y).*x_j)

J = [];
count = 1;
while(1)
Jcost(1) = 1/(2*len)*sum((x*t-y).^2);
for i=1:length(t)
delta = (1/len)*sum((x*t-y).*x(:,i));
ts(i) = t(i) - al*delta;
end

t = ts;
Jcost(2) = 1/(2*len)*sum((x*t-y).^2);

%% J(count) = 1/(2*len)*sum((t*x-y).^2);
count = count+1;
if(abs(Jcost(1)-Jcost(2))<1e-10)
break;
end
end


disp(1);
scatter(x(:,2),y); hold on;
y1 = x*t;

plot(x(:,2), y1); hold off;






