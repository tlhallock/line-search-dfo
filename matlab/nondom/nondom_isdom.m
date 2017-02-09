function [isdom, dominates] = nondom_isdom(nondom, newpoint)

isdom = false;

%if nargin < 2
%	ignore_idx = -1;
%end

for i = 1:size(nondom.set, 2)
%	if ignore_idx == i
%		continue;
%	end
	
	if nondom_dominates(nondom.set(:,i), newpoint)
		isdom = true;
		dominates = nondom.set(:,i);
		return;
	end
end





end
