clear all

y = [0.5 1 2 0];
x = [1 2 4 0];

t0 = 0;
t1 = 0;
al = 0.01;

%% cost function J(t0, t1) = 1/2m * sum((t0+t1x - y)^2)

i = 1;
while(1)
    a = t0 - al*(1/9)*sum(t0+t1*x-y);
    b = t1 - al*(1/9)*sum((t0+t1*x-y).*x);

    if(abs(a-t0)<1e-10 && abs(b-t1)<1e-10 )
        break;
    end
    t0 = a;
    t1 = b;
i = i+1;
end

disp(i);
scatter(x,y); hold on;
y1 = t0+t1*x;

plot(x, y1); hold off;

