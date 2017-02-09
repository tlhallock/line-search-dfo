function [nondom] = nondom_create()

nondom = struct();
nondom.set = [];
nondom.add = @(x)(nondom_add(nondom, x));
nondom.isdom = @(x)(nondom_isdom(nondom, x));

end
