% Codes for CVPR-15 work `Face Alignment by Coarse-to-Fine Shape Searching'
% Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
% Released on July 25, 2015

% Grab bboxes and frame numbers

clear;

file_id = fopen('data/bbox_annotations/SWC024_childface_5000frames.txt');
frames_bboxes = textscan(file_id, '%s%f%f%f%f%*[^\n]', 'delimiter', ' ');
fclose(file_id);
nameList_temp = frames_bboxes(:,1);
nameList_temp = nameList_temp{1,1};
bbox_temp = frames_bboxes(:,2:end);

% Save images from video to file
video_filename = 'data/videos/SWC024_2016_06_27_pivothead_AVI.avi';
video = VideoReader(video_filename);

'test0'
already_exist = true;

n_frames = 5000;
if ~already_exist
    for i=1:n_frames
      frame=readFrame(video);
      thisfile=sprintf('output/tmp/%d.jpg',i);
      imwrite(frame,thisfile);
    end
    'test1'
end

% Modify namelist to have .jpg
% TODO this is really slow..
if exist('nameList') ~= 1
    nameList = {};
    for i=1:length(nameList_temp)
        nameList{i,1} = [nameList_temp{i} '.jpg'];
    end
end

'test2'
% Set image root and load models
img_root = './output/tmp/';
load ./model/mean_simple_face.mat mean_simple_face;
load ./model/target_simple_face.mat target_simple_face;
load ./model/CFSS_Model_0.mat priorModel testConf model;

% This code is for OMRON bbox annotations
% Which are square and the format is NA, x1, y1, x2
x1 = bbox_temp{1,2};
y1 = bbox_temp{1,1};
x2 = bbox_temp{1,4};
y2 = bbox_temp{1,3};
bbox = [y1 y2 x1 x2];

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