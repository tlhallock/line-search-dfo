function [nondom] = nondom_add(nondom, newpoint)

if nondom_isdom(nondom, newpoint)
	disp('is dominated');
	return;
end

idx = 1;
while idx <= size(nondom.set, 2)
	if nondom_dominates(newpoint, nondom.set(:,idx))
		nondom.set(:, idx) = [];
		disp('dominates');
	else
		idx = idx + 1;
	end
end

nondom.set = [nondom.set , newpoint];

end
