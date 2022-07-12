
clear all;

addpath('C:\Users\Biolab\Desktop\eeglab2019_1');
eeglab;
close all;

cd f:\

load meg_important_ch_idx

cd('X:\#1 개인폴더\SHLee\MEG_freq\')

for i = 1: 31
    mkdir(['Subs' sprintf('%.3d',i)]);
end

cd('X:\#1 개인폴더\SHLee\MEG_freq_log\')
for i = 1: 31
    mkdir(['Subs' sprintf('%.3d',i)]);
end

for i = 1: 31
    mkdir(['Subs' sprintf('%.3d',i)]);
end
% 
% cd('X:\#1 개인폴더\SHLee\MEG_freq\');
% 
% fileList = dir;
% fileList = fileList(3:end);



for i = 1:31
    cd F:\MEG_data
    number = sprintf('%.3d',i);
    load(['data_BD', number, '.mat']);
    load(['label_BD', number,'.mat']);
    
    meg_data_data = meg_data_data(:,locIdx,:);
    fs = 1024;
    
    xlen = size(meg_data_data,3);                   % signal length
    t = (0:xlen-1)/fs;
    
    % make directory (class)
    cd(['X:\#1 개인폴더\SHLee\MEG_freq\' 'Subs' sprintf('%.3d',i)])
    for folders = 1: 10
        mkdir(num2str(folders));
    end
    cd(['X:\#1 개인폴더\SHLee\MEG_freq_log\' 'Subs' sprintf('%.3d',i)])
    for folders = 1: 10
        mkdir(num2str(folders));
    end
    
    for trials = 1: size(meg_data_data,1)
       tfmap = zeros(size(meg_data_data,2),151,98);
       tfmapLog = zeros(size(meg_data_data,2),151,98);
        
       parfor chs = 1: size(meg_data_data,2)
            
            x =  squeeze(meg_data_data(trials,chs,:));
%             %[ersp,itc,powbase,times,freqs,erspboot,itcboot,itcphase] = timef(x',[], [],fs,0.1, 'maxfreq', 50, 'plotersp', 'off', 'plotitc', 'off');
%             wlen = 32;                        % window length (recomended to be power of 2)
%             nfft = 4*wlen;                      % number of fft points (recomended to be power of 2)
%             hop = wlen/16;                       % hop size
% 
%             TimeRes = wlen/fs;                  % time resulution of the analysis (i.e., window duration), s
%             FreqRes = 2*fs/wlen;                % frequency resolution of the analysis (using Hanning window), Hz
% 
%             % time-frequency grid parameters
%             TimeResGrid = hop/fs;               % time resolution of the grid, s
%             FreqResGrid = fs/nfft;              % frequency resolution of the grid, Hz 
% 
%             % perform STFT
%             w1 = wlen; %hanning(wlen, 'periodic');
%             figure();
%             [~, fS, tS, PSD] = spectrogram(x, w1, wlen-hop, [], fs,'yaxis');
%             Samp = 20*log10(sqrt(PSD.*enbw(w1, fs))*sqrt(2));
            
            % Window duration (in seconds):
           dur = 0.05;
           % Spectrogram settings (in samples):
           winSize = round(fs*dur);
           overlap = round(winSize*0.8);
           fftSize = winSize*30;
           % Plot the spectrogram:
           [~, fS, tS, PSD]= spectrogram(x,winSize,overlap,fftSize,fs,'yaxis');
           Samp = 10*log10(sqrt(PSD.*enbw(winSize, fs))*sqrt(2));
           % spectrogram(x,winSize,overlap,fftSize,fs,'yaxis');
            % perform spectral analysis
%             w2 = xlen;%hanning(xlen, 'periodic');
%             [PS, fX] = periodogram(x, w2, nfft, fs, 'power');
%             Xamp = 20*log10(sqrt(PS)*sqrt(2));
%             
           tfmap(chs,:,:) = PSD(1:151,:);
           tfmapLog(chs,:,:) = Samp(1:151,:);
           PSD = [];
           Samp = [];
       end
       
       save(['X:\#1 개인폴더\SHLee\MEG_freq\' 'Subs' sprintf('%.3d',i) '\' num2str(meg_data_label(trials)),'\tfmap_sub' sprintf('%.3d',i) '_trial' sprintf('%.4d',trials)],'tfmap');
       save(['X:\#1 개인폴더\SHLee\MEG_freq_log\' 'Subs' sprintf('%.3d',i) '\' num2str(meg_data_label(trials)),'\tfmap_sub' sprintf('%.3d',i) '_trial' sprintf('%.4d',trials)],'tfmapLog');
       
    end
    disp(i)
    
end
disp('done');


%% test
cd 'X:\#1 개인폴더\SHLee\MEG_freq\Subs030\1'
figure();
a = dir;
a = a(3:end);

for files = 1:length(a)
    load(a(files).name)
    figure();
    for j = 1: size(tfmap,1)
        
        subplot(5,5,j)
        imagesc(squeeze(tfmap(j,:,:)));
    end
end