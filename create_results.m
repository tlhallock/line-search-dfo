function [results] = create_results()


results = struct();
results.iters = 0;
results.restorations = 0;
results.ftype_iterations = 0;
results.eightab_iterations = 0;
results.filter_rejected_iterations = 0;
results.x_optimal = 0;
results.f_min = 0;
results.filter = nondom_create();


end
