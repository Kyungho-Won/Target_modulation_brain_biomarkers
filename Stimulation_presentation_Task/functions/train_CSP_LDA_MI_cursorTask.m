% train_CSP_LDA_MI_cursorTask.m
% train CSP and LDA with calibration data and save CSP and LDA for test
clc; clear;
cd('C:\Users\BioComput_HGKim\Desktop\khwon\Stim_for_MI');
nbrun = 4;
frame = [0.5 3.5];
bp = [8 30];
DSI_locs = readlocs('DSI_locs_for_openViBE.locs');
eeg_left3D = [];
eeg_right3D = [];

% pre-defined left/right event marker
left_marker = 1;
right_marker = 2;

[baseName, folder] = uigetfile('*.*', 'MultiSelect','on');
fnames_for_train = fullfile(folder, baseName);

% step1. Load, pre-procesing data
for nRun = 1:length(fnames_for_train)
    fname = fnames_for_train{nRun};
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
        eeg_left = cat(3, eeg_left, eeg.data(:, MI_left(nTrial)+frame(1)*eeg.srate:MI_left(nTrial)+(frame(2))*eeg.srate-1));
        eeg_right = cat(3, eeg_right, eeg.data(:, MI_right(nTrial)+frame(1)*eeg.srate:MI_right(nTrial)+(frame(2))*eeg.srate-1));
    end
    
    eeg_left3D = cat(3, eeg_left3D, eeg_left);
    eeg_right3D = cat(3, eeg_right3D, eeg_right);
end

% figure,
% subplot(2,1,1);
% spectopo(eeg_left3D, 0, eeg.srate); xlim([0 50]); title('Left MI');
% pbaspect([1 1 1]);
% subplot(2,1,2);
% spectopo(eeg_right3D, 0, eeg.srate); xlim([0 50]); title('Right MI');
% pbaspect([1 1 1]);

% step2. train CSP and LDA
% now x1 and x2 are [ch x time x trials]
% ch_list = 1:9;
% % eeg_left3D([14 18], :, :) = [];
% % eeg_right3D([14 18], :, :) = [];
% eeg_left3D = eeg_left3D(1:9, :, :);
% eeg_right3D = eeg_right3D(1:9, :, :);
% ch_list =1:9;
CSP_trained = extract_CSP(eeg_left3D, eeg_right3D);
LDA_trained = train_FLDA(eeg_left3D, eeg_right3D, CSP_trained.W, 2);
disp(LDA_trained)
ch_list = 1:19;
zcsp = zeros(19, 2);
zcsp(ch_list, 1) = CSP_trained.W(:,1);
zcsp(ch_list, 2) = CSP_trained.W(:,end);

figure,
subplot(1,2,1); topoplot(zcsp(:,1), eeg.chanlocs); title('Left', 'fontsize', 14);
subplot(1,2,2); topoplot(zcsp(:,end), eeg.chanlocs); title('right', 'fontsize', 14);

% figure,
% subplot(1,2,1); topoplot(CSP_trained.W(:,1), eeg.chanlocs); title('Left', 'fontsize', 14);
% subplot(1,2,2); topoplot(CSP_trained.W(:,end), eeg.chanlocs); title('right', 'fontsize', 14);
% 
% % step3. save the trained CSP and LDA weight
train_mdl = [];
train_mdl.CSP = CSP_trained;
train_mdl.mdl = LDA_trained;
% 
% % path_out = uigetdir;
path_out = uiputfile;
% % fname_out = [path_out, '\mdl_trained.mat'];
fname_out = path_out;
save(fname_out, '-struct', 'train_mdl', '-v7.3');

%% subfunctions

function out = rm_unused(eeg, rm_labels)
rm_idcs = ismember({eeg.chanlocs.labels}, rm_labels);
eeg.data = eeg.data(~rm_idcs, :);
eeg.chanlocs = eeg.chanlocs(~rm_idcs);

out = eeg;
end