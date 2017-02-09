
nondom = nondom_create();

nondom = nondom_add(nondom, [1;2;3]);
nondom
nondom = nondom_add(nondom, [3;2;1]);
nondom
nondom = nondom_add(nondom, [3;1;3]);
nondom

nondom_isdom(nondom, [3; 1; 1]) # Should be false
nondom_isdom(nondom, [1; 1; 1]) # Should be false
nondom_isdom(nondom, [3; 2; 3]) # Should be true
nondom_isdom(nondom, [5; 5; 5]) # Should be true
nondom_isdom(nondom, [1; 2; 3]) # Should be true

nondom = nondom_add(nondom, [1;1;1]);

nondom



