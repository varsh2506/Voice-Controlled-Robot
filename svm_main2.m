%% Voice Control

% folder='C:\Users\malek\Documents\MATLAB'
% audio_files=dir(fullfile(folder,'*.m'))
% for k=1:numel(audio_files)
%   filename=audio_files(k).name
%   %Do what you want with filename
%   % create a new file name 'new_file' for example
%   folder_destination='C:\Users\malek\Documents'  % for example
%   file_dest=fullfile(folder_destination,'new_file'
%   % .....
% end

% folder="A:\MyFiles\Machine Learning\projects\Voice Controlled Bot\data\right";
% af=dir(fullfile(folder,"*.wav"));
% 
% c=[];
% F=[];
% m=0;
% 
% for k=1:numel(af)
%    filename=af(k).name;
%    
%    disp(filename);
%    
%    [aud,f]=audioread(filename);
%    [aud]=mfcc(aud,f);
%    
% %    if size(aud,1)*size(aud,2)>m
% %     m=size(aud);
% %    end
%    
%    aud=reshape(aud,1,size(aud,1)*size(aud,2));
%    aud=[aud zeros(1,(98*14)-size(aud,1)*size(aud,2))];
%    
%    c=[c;aud];
% end

%% Loading data set

load("svm_go.mat");
X=c;
y=ones(size(X,1),1);
load("svm_left.mat");
X=[X;c];
y=[y;2*ones(size(c,1),1)];
load("svm_right.mat");
X=[X;c];
y=[y;3*ones(size(c,1),1)];
load("svm_stop.mat");
X=[X;c];
y=[y;4*ones(size(c,1),1)];

p=isinf(X);
X(find(p==1))=0;

randidx=randperm(size(X,1));

X=[ones(size(X,1),1) X];

X=X(randidx,:);
y=y(randidx,1);

Xtrain=X(1:6000,:);
ytrain=y(1:6000,1);

Xcv=X(6001:8196,:);
ycv=y(6001:8196,1);

Xtest=X(8197:end,:);
ytest=y(8197:end,1);

%% Initializing 

m=size(Xtrain,1);
n=size(Xtrain,2);

lambda=1;
theta=zeros(n,1);


%% SVM

% C=100;
% sigma=1;
[model]=svmtrain(ytrain,Xtrain);
fprintf("********************************************\n");
[p]=svmpredict(ytrain,Xtrain,model);

fprintf("Accuracy: %f\n",double(mean(p==ytrain))*100);

%% SVM with ones vs all

% [go]=svmtrain(double(ytrain==1),Xtrain);
% fprintf("************************\n");
% [p]=svmpredict(double(ycv==1),Xcv,go);
% 
% fprintf("Accuracy: %f\n",double(mean(p==double(ycv==1)))*100);
% 
% [left]=svmtrain(double(ytrain==2),Xtrain);
% fprintf("************************\n");
% % [p]=svmpredict(double(ycv==2),Xcv,left);
% % 
% % fprintf("Accuracy: %f\n",double(mean(p==double(ycv==2)))*100);
% 
% [right]=svmtrain(double(ytrain==3),Xtrain);
% fprintf("************************\n");
% 
% [p]=svmpredict(double(ycv(2:3,1)==1),Xcv(2:3,:),go)
% [p]=svmpredict(double(ycv(2:3,1)==2),Xcv(2:3,:),left)
% [p]=svmpredict(double(ycv(2:3,1)==3),Xcv(2:3,:),right)

