clear all

y = [0.5 1 2 0];
x = [1 2 4 0];

m = length(x);
%% cost function J(t0, t1) = 1/2m * sum((t0+t1x - y)^2)

val = y(find(x==0));
t0 = y-1:0.1:y+1;
len = length(t0);
t1 = linspace(-1, 1, len);

Jc = eps;
t0c = t0(1);
t1c = t1(1);

for i=1:len,
    for j=1:len,
        temp = 1/(2*m) * sum((t0(i)+t1(j)*x-y).^2);
        if(i == 1 && j==1)
            Jc = temp;
            t0c = t0(i);
            t1c = t1c(j);
        else
            if(temp < Jc)
                Jc = temp;
                t0c = t0(i);
                t1c = t1(j);
            end
        end
    end
end



disp(i);
scatter(x,y); hold on;
y1 = t0c+t1c*x;

plot(x, y1); hold off;

