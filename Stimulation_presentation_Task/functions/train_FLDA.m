function C = train_FLDA(x1_train, x2_train, W_CSP, nb_filter, varargin)
% C = train_FLDA(xTr, yTr, W_CSP)
% input:
%    x1_train: training data for class 1 [ch, time, trial]
%    x2_train: training data for class 2 [ch, time, trial]
%    W_CSP: CSP filters [ch, ch]
%    xTr:
%    nb_filter: the number of CSP filters (first n, last n filters)

% return error if x1_train and x2_train have different dimension
if size(x1_train) ~= size(x2_train)
    error('x1_train and x2_train have different dimension.');
end

nCh = size(W_CSP, 2);
nTrial_x1 = size(x1_train, 3);
nTrial_x2 = size(x2_train, 3);
csp_pat = [1:nb_filter, nCh-(nb_filter)+1:nCh];

feature_x1 = [];
feature_x2 = [];
% step1. filtering the data with CSP filter
for i=1:nTrial_x1
    cur_featx1 = log(var(W_CSP(:, csp_pat)'*x1_train(:,:,i), [], 2));
    feature_x1 = cat(2, feature_x1, cur_featx1);
    
end

for i=1:nTrial_x2
    cur_featx2 = log(var(W_CSP(:, csp_pat)'*x2_train(:,:,i), [], 2));
    feature_x2 = cat(2, feature_x2, cur_featx2);
end
% feature_x1, x2: [feature x trial]

% step2. calculate FLDA w, w0 from the features
disp(size(feature_x1));
disp(size(feature_x2));
[w, w0] = sub_flda(feature_x1, feature_x2);

% evaluate training accuracy (just for checking)
y_x1 = [];
y_x2 = [];
for j=1:size(feature_x1, 2)
    y_x1(j) = w'*feature_x1(:,j)+w0;
end

for j=1:size(feature_x2, 2)
    y_x2(j) = w'*feature_x2(:,j)+w0;
end


% fitcsvm
svm_train_x1 = feature_x1';
svm_train_x2 = feature_x2';

label_x1 = ones(size(svm_train_x1,1), 1);
label_x2 = -ones(size(svm_train_x2,1), 1);

SVMmdl = fitcsvm([svm_train_x1;svm_train_x2], [label_x1;label_x2], ...
    'KernelFunction', 'rbf', 'ClassNames', [-1 1]);

train_predict = predict(SVMmdl, [svm_train_x1;svm_train_x2]);
train_hit = length(find(train_predict==[label_x1;label_x2]));


hit = length(find(y_x1>0)) + length(find(y_x2<0));

C.hit = hit;
C.lda_W = w;
C.lda_w0 = w0;
C.y_x1 = y_x1;
C.y_x2 = y_x2;

C.svm_mdl = SVMmdl;
C.svm_predict = train_predict;
C.svm_hit = train_hit;
end
%% SUB FUNCTION
function [w,w0] = sub_flda(class1,class2)
% function [w,w0] = sub_flda(class1,class2)
% Implemented as written in
% 'http://en.wikipedia.org/wiki/Linear_discriminant_analysis'
% 'Fisher's linear discriminant' section
%
% *** INPUT ***
% class1 : [n_feat x n_trials]
% class2 : [n_feat x n_trials]
%
% *** OUTPUT ***
% w : [n_feat x 1] weight for LDA
% w0 : [scalar] bias
%
% *** MODIFICATION ***
% 2010.08.19 : Minkyu Ahn
% - first written
%
%--------------------------------------------------------------------------
% Minkyu Ahn        frerap@gist.ac.kr
% http://biocomput.gist.ac.kr
%

%------------------------------------------------
% As wiki
n_feat = size(class1,1);
n_trials = [size(class1,2), size(class2,2)];

% Mean zero
m1=mean(class1')'; % [1 x n_feat]
m2=mean(class2')'; % [1 x n_feat]

% Baseline correction for each feature
for i=1:n_trials(1)
    class1(:,i)=class1(:,i)-m1;
end
for i=1:n_trials(2)
    class2(:,i)=class2(:,i)-m2;
end

% Corvariances
S1 = class1 * class1'; % [n_feat  x  n_feat]
S2 = class2 * class2'; % [n_feat  x  n_feat]

% W for FLDA
% w=inv(S1+S2)*(m1-m2);  % [n_feat  x  1]
w = (S1+S2)\(m1-m2);

% W0 for FLDA
w0=-w'*((m1+m2)/2);
end

