%% Load dataset Kim(2019)
% Dataset: K.-T. Kim, H.-I. Suk, and S.-W. Lee, “Commanding a Brain-Controlled Wheelchair using Steady-State Somatosensory Evoked Potentials,” IEEE Trans. on Neural Systems & Rehabilitation Engineering, Vol. 26, No. 3, 2018, pp. 654-665.
% made by Daeun Gwon in BCI LAB of HGU, South Korea

clear; clc;
filtering_freqband = [ ]; % filtering range [low high] ex) [1 10]
fpath = ''; % path of dataset dir
time = [-2 2]; % window size(second)

for subjectNum = [1:7 9:12] 
    display(['Load subject ' num2str(subjectNum)]);
    eeg = pop_loadbv(fpath,dataname);   % public dataset file format is vhdr

    data = eeg.data;    srate = eeg.srate;
    event = squeeze(struct2cell(eeg.event));  event = event(:,3:end);
    event_latency = cell2mat(event(1,:));   event_type = event(7,:); 
    event_type = string(event_type);
    
    index_left = find(event_type == 'S  1');
    index_right = find(event_type == 'S  2');
    
    % Referencing(common average)
    rdata = mean(data);
    rdata = repmat(rdata,size(data,1),1);
    data = data - rdata;
    data = ft_preproc_bandpassfilter(data, srate,filtering_freqband,4,'but');
 
    preprocessed_left_signal =[];
    for trial = 1:length(index_left)
        tmp_data = data(:,index_left(trial)+1+srate*time(1):index_left(trial)+srate*time(2)) ;
        tmp_data = tmp_data-mean(tmp_data,2); 
        preprocessed_left_signal(:,:,trial) =  tmp_data';
    end

    preprocessed_right_signal =[];
    for trial = 1:length(index_right)
        tmp_data = data(:,index_right(trial)+1+srate*time(1):index_right(trial)+srate*time(2)) ;
        tmp_data = tmp_data-mean(tmp_data,2); 
        preprocessed_right_signal(:,:,trial) =  tmp_data';
    end

end
