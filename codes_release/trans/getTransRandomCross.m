% Codes for CVPR-15 work `Face Alignment by Coarse-to-Fine Shape Searching'
% Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
% Released on July 25, 2015

function T = getTransRandomCross(pose)
% T = getTransRandomCross(pose)
% produce trans to project poese onto randomly selected another pose
% size(pose) = [m,2n], each row must in the format of X1...Xn,Y1...Yn.
% T is a array containing n structures. Each one is the trans for one
% sample.

n = size(pose,2);
if mod(n,2),error('size(pose,2) is odd!');end
n = n / 2;
m = size(pose,1);
pr = randi(m,m,1);
I = find(pr-[1:m]'==0);
while ~isempty(I)
    pr(I) = randi(m,length(I),1);
    I = find(pr-[1:m]'==0);
end
if (false)
    T = arrayfun(@(i)cp2tform(reshape(pose(i,:),n,2),reshape(pose(pr(i),:),n,2),'nonreflective similarity'),1:m);
    if m>=100
        warning('Please launch matlabpool to speed up your program!');
    end
else
    T = cell(m,1);
    newPose = pose(pr,:);
    parfor i = 1:m
        T{i} = cp2tform(reshape(pose(i,:),n,2),reshape(newPose(i,:),n,2),'nonreflective similarity');
    end
    T = cell2mat(T);
end

end

