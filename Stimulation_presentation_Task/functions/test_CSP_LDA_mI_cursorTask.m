clc; clear;
nbrun = 4;
frame = [0.5 3.5];
bp = [8 30];
DSI_locs = readlocs('DSI_locs_for_openViBE.locs');
eeg_left3D = [];
eeg_right3D = [];

% pre-defined left/right event marker
left_marker = 1;
right_marker = 2;

% step1. Load, pre-procesing data
for nRun = 1
    fname = sprintf('./dat/intern/Cursor-test-run%02d.gdf', nRun);
    eeg = pop_biosig(fname);
    event_type = [eeg.event.type];
    event_latency = [eeg.event.latency];
    
    % rm unused ch
    eeg = rm_unused(eeg, {'X1', 'X2', 'X3', 'A2'});
    eeg.chanlocs = DSI_locs;
    
    MI_left = event_latency(event_type==left_marker);
    MI_right = event_latency(event_type==right_marker);
    
    % bandpass filtering and bandstop filtering (58-60Hz)
    eeg.data = ft_preproc_bandpassfilter(eeg.data, eeg.srate, bp, 4, 'but');
    eeg.data = ft_preproc_bandstopfilter(eeg.data, eeg.srate, [58 60], 4, 'but');
    
    % plotting time signal data
    %     eegplot(eeg.data, 'srate', eeg.srate, 'winlength', 10, 'eloc_file', eeg.chanlocs);
    
    % data segmentation [0 4000]ms to the stimulus onset
    eeg_left = [];
    eeg_right = [];
    for nTrial = 1:length(MI_left) % they are balanced (left/right MI)
        eeg_left = cat(3, eeg_left, eeg.data(:, MI_left(nTrial)+frame(1)*eeg.srate:MI_left(nTrial)+(frame(2)-frame(1))*eeg.srate-1));
        eeg_right = cat(3, eeg_right, eeg.data(:, MI_right(nTrial)+frame(1)*eeg.srate:MI_right(nTrial)+(frame(2)-frame(1))*eeg.srate-1));
    end
    
    eeg_left3D = cat(3, eeg_left3D, eeg_left);
    eeg_right3D = cat(3, eeg_right3D, eeg_right);
end

% step2. train load and test LDA
[file_mdl, path_mdl] = uigetfile;
train_mdl = load([path_mdl file_mdl]);
disp(train_mdl);
% now x1 and x2 are [ch x time x trials]
% eeg_left3D = eeg_left3D(1:9, :, :);
% eeg_right3D = eeg_right3D(1:9, :, :);
nCh = size(eeg_left3D, 1);
Params = struct('n_filter', 2);
label_all = [ones(size(eeg_left3D,3),1);-ones(size(eeg_left3D,3),1)];
eeg_all = cat(3, eeg_left3D, eeg_right3D);
csp_pat = [1:Params.n_filter, nCh-(Params.n_filter)+1:nCh];

cur_predict = [];
for j=1:size(eeg_all, 3)
    cur_feat = log(var(train_mdl.CSP.W(:, csp_pat)'*eeg_all(:,:,j), [], 2));
    
    lda_predict(j) = train_mdl.mdl.lda_W'*cur_feat+train_mdl.mdl.lda_w0;
    svm_predict(j) = predict(train_mdl.mdl.svm_mdl, cur_feat');
end
lda_predict = 2* ((lda_predict>0)-0.5);

lda_acc = sum(label_all' == lda_predict);
svm_acc = sum(label_all' == svm_predict);

fprintf('LDA hit: %02d out of %02d, acc: %.2f\n', lda_acc, size(eeg_all, 3), lda_acc/size(eeg_all,3));
fprintf('SVM (rbf) hit: %02d out of %02d, acc: %.2f\n', svm_acc, size(eeg_all, 3), svm_acc/size(eeg_all,3));

%% subfunctions

function out = rm_unused(eeg, rm_labels)
rm_idcs = ismember({eeg.chanlocs.labels}, rm_labels);
eeg.data = eeg.data(~rm_idcs, :);
eeg.chanlocs = eeg.chanlocs(~rm_idcs);

out = eeg;
end










