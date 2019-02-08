
L = 2


f1 = @(x)(1);
f2 = @(x)(x(1));
f3 = @(x)(x(2));
f4 = @(x)(x(1) * x(1));
f5 = @(x)(x(1) * x(2));
f6 = @(x)(x(2) * x(2));

f = @(x)([f1(x),f2(x),f3(x),f4(x),f5(x),f6(x)]);
original = @(x)(5 + x(1) + x(2) + (x(1) + x(2)) ** 2 - x(2) ** 4 / 1000 );

Y = [linspace(-L, L, 6)',  linspace(-L, L, 6)'] + rand(6, 2) / 10;
%Y(1, 1) = -L
%Y(1, 2) = -L

%Y = [sin(linspace(0, 2*pi, 6))', cos(linspace(0, 2*pi, 6))']
Y(6,:) = 0





funcValues = zeros(1,6)
for i = 1:size(Y, 1)
    funcValues(i) = original(Y(i,:));
end

V = zeros(6, 6);
for i = 1:6
    V(i, :) = f(Y(i, :));
end
phi = V\funcValues'

p = [1, 2]
f(p)*phi
original(p)

for i = 1:6
    f(Y(i, :))*phi - original(Y(i, :))
end


tx = ty = linspace (-L, L, 41)';
[xx, yy] = meshgrid (tx, ty);
tz1 = zeros(size(tx, 1), size(ty, 1));
for i = 1:size(tx, 1)
    for j = 1:size(ty, 1)
        tz1(i, j) = original([tx(i), ty(j)]);
    end
end

tz2 = zeros(size(tx, 1), size(ty, 1));
for i = 1:size(tx, 1)
    for j = 1:size(ty, 1)
        tz2(i, j) = f([tx(i), ty(j)])*phi;
    end
end

t = linspace(-L, L, 10);

mesh(tx, ty, tz1);
hold on
mesh(tx, ty, tz2);


for i = 1:size(Y, 1)
    plot(Y(i, 1), Y(i, 2), 'r*');
end



plot(t, t + linspace(0,  0.25, 10) )
plot(t, t + linspace(0, -0.25, 10) )
