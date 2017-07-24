n = 50;
h = zeros(n, 1, 2);
h(:, 1, 1) = rand(n, 1) / 10 + exp(linspace(10, 1, n))';
h(:, 1, 2) = rand(n, 1) / 10 + exp(linspace(20, .5, n))';
data_profile(h, 1, 1);

