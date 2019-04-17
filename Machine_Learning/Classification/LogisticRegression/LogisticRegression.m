clear all
%%/Users/rafiul_rasel/Desktop/Machine_Learning/Classification/LogisticRegression



y = [0; 0; 0; 0; 1; 1; 1; 1; 1; 1];
m = length(y);
x = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1;];
x = [x, [0; 1; 2; 3; 4; 5; 6; 7; 8; 9]];

t = zeros(1,length(x(1,:))).';
ts = t;
al = 1;

count = 1;

%% sigmoid function h_theta = 1/(1+exp(x*theta));
%% cost(h_theta,y) = -y*log(h_theta)-((1-y)*log(1-h_theta))
%% J(theta) = (1/m)*sum_(i=1,m)(cost(h_theta,y))
%% theta_j = theta_j-alpha*(1/m)*sum(h_theta - y)*x;

count = 1;
while(1)
    [J(count) delta] = costFunctionLR(x, y, t);
    t = t - al*delta;

    if(count == 1000)
        break;
    end
    count = count+1;
end


disp(1);
%%scatter(x(:,2),y); hold on;

x1 = [1 3; 1 3.2; 1 3.4; 1 3.6; 1 3.8];
h_theta = 1./(1+exp(-x1*t));

y1 = round(h_theta);

plot(x1(:,2), y1, 'rx', 'markersize', 10); hold off;











