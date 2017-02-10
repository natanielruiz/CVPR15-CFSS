% Codes for CVPR-15 work `Face Alignment by Coarse-to-Fine Shape Searching'
% Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
% Released on July 25, 2015

clear;

% Grab image names and bounding boxes
file_id = fopen('data/test_csv_name_bbox.csv');
filename_bboxes = textscan(file_id, '%s%f%f%f%f', 'delimiter', ',');
fclose(file_id);
nameList_temp = filename_bboxes(:,1);
nameList = nameList_temp{1,1};
bbox_temp = filename_bboxes(:,2:end);
bbox = [bbox_temp{1,1} bbox_temp{1,2} bbox_temp{1,3} bbox_temp{1,4}];

% Set image root and load models
img_root = './imageSource/';
load ./model/mean_simple_face.mat mean_simple_face;
load ./model/target_simple_face.mat target_simple_face;
load ./model/CFSS_Model.mat priorModel testConf model;

m = length(nameList);
mt = size(model{1}.tpt,1);
T = cell(1,testConf.stageTot);
images = cell(m,1);

for level = 1:testConf.stageTot
    % 61. Re-trans
    if level == 1
        [images,T{level}] = testingsetGeneration(img_root,nameList,bbox,...
            priorModel,testConf.priors,mean_simple_face,target_simple_face);
        Pr = 1/mt * ones(m,mt);
    end;
    
    % 62. from Pr to sub-region center 
    currentPose = inferenceReg(images,model,Pr,level,testConf.regs);
    
    if level >= testConf.stageTot, break; end;
    
    T{level+1} = getTransToSpecific(currentPose,priorModel.referenceShape);
    images = transImagesFwd(images,T{level+1},testConf.win_size,testConf.win_size);
    currentPose = transPoseFwd(currentPose,T{level+1});
    
    % 63. from sub-region center to Pr
    Pr = inferenceP(images,model,currentPose,level,testConf.probs);
end;

estimatedPose = currentPose;
for level = testConf.stageTot:-1:1
    estimatedPose = transPoseInv(estimatedPose,T{level});
end;

% Write landmarks to csv
% Output format is:
% X1,X2,...,Xn,Y1,Y2,...,Yn
csvwrite('output/test_landmarks.csv', estimatedPose);