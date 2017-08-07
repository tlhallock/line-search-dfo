n = 50;
h = zeros(n, 2, 2);
h(:, 1, 1) = rand(n, 1) / 10 + linspace(10, .5, n)';
h(:, 1, 2) = rand(n, 1) / 10 + linspace(20, .5, n)';
h(:, 2, 1) = rand(n, 1) / 10 + linspace(15, .5, n)';
h(:, 2, 2) = rand(n, 1) / 10 + linspace(20, .5, n)';
# data_profile(h, ones(2), 1);

perf_profile(h, .001, 0);