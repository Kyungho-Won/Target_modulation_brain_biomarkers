%% Load dataset Cho(2017)
% Dataset: H. Cho, M. Ahn, S. Ahn, K. Kwon, and S. C. Jun, “EEG Datasets for Motor Imagery Brain-Computer Interface,” GigaScience, Vol. 6, No. 1, 2017, pp. 1-8.
% made by Daeun Gwon in BCI LAB of HGU, South Korea

clear; clc;
filtering_freqband = [ ]; % filtering range [low high] ex) [1 10]
fpath = ''; % path of dataset dir
time = [-2 2]; % window size(second)
for subjectNum = 1:52 
    display(['Load subject ' num2str(subjectNum)]);
    load([fpath int2str(subjectNum) '.mat']); % public dataset file format is mat

    data = eeg.raw_left;    event = eeg.event;  
    srate = eeg.srate;      window_size = time(2)-time(1);
    
    % Referencing(common average)
    rdata = mean(data);
    rdata = repmat(rdata,size(eeg.raw_left,1),1);
    data = data - rdata;
    data = reformsig(data',eeg.n_trials); % [ch x t] => [t x ch x trials]
    
    preprocessed_left_signal = zeros(size(eeg.raw_left,1),srate*window_size,eeg.n_trials);
    for trial = 1:eeg.n_trials
        tmp_data = ft_preproc_bandpassfilter(squeeze(data(:,:,trial))', srate, filtering_freqband,4,'but');
        % baseline correction
        tmp_data = tmp_data(:,1:512*window_size) - mean(tmp_data(:,1:512*window_size) ,2);
        preprocessed_left_signal(:,:,trial) =  tmp_data;
    end

    data = eeg.raw_right;
    %referencing(common average)
    rdata = mean(data);
    rdata = repmat(rdata,size(eeg.raw_right,1),1);
    data = data - rdata;
    data = reformsig(data',eeg.n_trials); % [ch x t] => [t x ch x trials]
    
    preprocessed_left_signal = zeros(size(eeg.raw_right,1),srate*window_size,eeg.n_trials);
    for trial = 1:eeg.n_trials
        tmp_data = ft_preproc_bandpassfilter(squeeze(data(:,:,trial))', srate, filtering_freqband,4,'but');
        % baseline correction
        tmp_data = tmp_data(:,1:512*window_size) - mean(tmp_data(:,1:512*window_size) ,2);
        preprocessed_right_signal(:,:,trial) =  tmp_data;
    end
end