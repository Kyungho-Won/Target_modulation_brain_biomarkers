
function out = train_CSPFLDA_mdl(target_axesA,target_axesB, Params, fnames)
% train_CSP_LDA_MI_cursorTask.m
% train CSP and LDA with calibration data and save CSP and LDA for test

DSI_locs = readlocs('DSI_locs_for_openViBE.locs');
frame = Params.frame;
bp = Params.bp;
% pre-defined left/right event marker
left_marker = Params.Markers_defined(1);
right_marker = Params.Markers_defined(2);
chCSP = Params.chCSP;
if iscell(fnames)
    total_runs = length(fnames);
elseif ischar(fnames)
    total_runs = 1;
end

% step1. Load, pre-procesing data
eeg_left3D = [];
eeg_right3D = [];
for nRun = 1:total_runs
    if iscell(fnames)
        fname = fnames{nRun};
    elseif ischar(fnames)
        fname = fnames;
    end   
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
    eeg.data = eeg.data(chCSP, :);
    
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

% outlier - greater than 100 micro volt
bad_left = find_badAmplitude(eeg_left3D, 100);
bad_right = find_badAmplitude(eeg_right3D, 100);
eeg_left3D(:, :, bad_left) = [];
eeg_right3D(:, :, bad_right) = [];

% step2. train CSP and LDA
% now x1 and x2 are [ch x time x trials]
CSP_trained = extract_CSP(eeg_left3D, eeg_right3D);
LDA_trained = train_FLDA(eeg_left3D, eeg_right3D, CSP_trained.W, Params.nCSP);
disp(LDA_trained);

zCSP_L = zeros(19,1);
zCSP_R = zeros(19,1);
zCSP_L(chCSP) = CSP_trained.W(:,1);
zCSP_R(chCSP) = CSP_trained.W(:,end);

axes(target_axesA);
topoplot(zCSP_L, eeg.chanlocs, ...
    'numcontour', 4, 'style', 'map');
title('Left', 'fontsize', 30, 'color', 'w');

axes(target_axesB);
topoplot(zCSP_R, eeg.chanlocs, ...
    'numcontour', 4, 'style', 'map');
title('right', 'fontsize', 30, 'color', 'w');

% step3. save the trained CSP and LDA weight
train_mdl = [];
train_mdl.CSP = CSP_trained;
train_mdl.mdl = LDA_trained;

path_out = uiputfile({'*.*'});
fname_out = path_out;
if ~contains(fname_out, '.mat')
    fname_out = strcat(fname_out, '.mat');
end
save(fname_out, '-struct', 'train_mdl', '-v7.3');

out = LDA_trained;
end
%% subfunctions

function out = rm_unused(eeg, rm_labels)
rm_idcs = ismember({eeg.chanlocs.labels}, rm_labels);
eeg.data = eeg.data(~rm_idcs, :);
eeg.chanlocs = eeg.chanlocs(~rm_idcs);

out = eeg;
end

function out = find_badAmplitude(data3D, threshold)
% data3D: ch x time x trial
out = [];
for n_trial = 1:size(data3D, 3)
    cur_trial = data3D(:, :, n_trial);
    if sum(max(abs(cur_trial), [], 2) > threshold)
        out = cat(1, out, n_trial);
    end
end

end