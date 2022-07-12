clear all
sub_str = {'S01','S02','S09','S18','S24','S28','S29'};
task_list = {'VS','VI', 'AS','AI'};


base_path = 'D:\EEG_data_Visual\VI\balanced\';

num_class = 9;
addpath('D:\Envision_EEG_TFdata\2D\eeglab2022.0');
result_path = 'D:\EEG_TFMap\VI\';
if ~isfolder(result_path)
    mkdir(result_path)
end

eeglab;
close all;

fs = 128;


for task = 2:2
    
    for sub =1: length(sub_str)
        disp([sub_str{sub} 'started']);
        tic
        % trial x channel x time
        
        a = load([base_path sub_str{sub} '\vi_data_balanced']);
        
        data = downsample(permute(a.balanced_vi_data,[3 2 1]),8);
        data = data(:,1:32,:);
        % time x channel x trial
        
        if num_class == 3
            label = a.balanced_vi_cate_label;
        else
            label = a.balanced_vi_label;
            label = label -1;
        end
        disp(unique(label))
        % trial x freq x time
        tfdata_ersp = zeros(length(label),32*10,200);
        % VI - 32*7,125
        tfdata_stft = zeros(length(label),32*6,122);
        
        tfdata_cwd = zeros(length(label),32*12,256);
        
        disp(['ERSP_ratio : ' num2str(size(tfdata_ersp,2)/size(tfdata_ersp,3))]);
        disp(['STFT_ratio : ' num2str(size(tfdata_stft,2)/size(tfdata_stft,3))]);
        disp(['CWD_ratio : ' num2str(size(tfdata_cwd,2)/size(tfdata_cwd,3))]);
        
        for trial = 1 : length(label)
           
            
            for channel = 1: 32
                % freq x time 
                % VI 512, [0 4000]
                ersp_result = newtimef(squeeze(data(:,channel,trial)),256,[0 2000],fs,0,'freqs',[0 50],'nfreqs',10,'plotitc','off', 'plotersp', 'off','verbose','off');
                figure();imagesc(ersp_result)
%                 newtimef(squeeze(data(:,channel,trial)),256,[0 2000],fs,0,'freqs',[0 50],'nfreqs',10)
                % VI 16, 12, 16
                [stft_result , F, T] = spectrogram(squeeze(data(:,channel,trial)), 14, 12, 14, fs);
% %  
% %                   tf=dtfd_nonsep(data(:,channel,trial),'cw',{30});
% figure();
% % %                   imagesc(tf')
% % %                   clf; vtfd(tf,data(:,channel,trial), fs);
% set(gca,'YDir','normal') 
% xlabel('Frequency (Hz)')
% ylabel('Time (s)');
% xtick
                % VI 2,16
                cwd_result = dec_dtfd_nonsep(data(:,channel,trial),'cw',{30},2,16);
                cwd_result = cwd_result';
%                 
                tfdata_ersp(trial,channel*10-9:channel*10,:) = ersp_result;
                tfdata_stft(trial,channel*6-5:channel*6,:) = stft_result(1:6,:);
                tfdata_cwd(trial,channel*12-11:channel*12,:) = cwd_result(1:12,:);
            end
            
            save_path_ersp = [result_path sub_str{sub} '\ERSP_' num2str(num_class) 'cl\' num2str(label(trial)) '\'];
            if ~isfolder(save_path_ersp)
                mkdir(save_path_ersp)
            end
            save_path_stft = [result_path sub_str{sub} '\STFT_' num2str(num_class) 'cl\' num2str(label(trial)) '\'];
            if ~isfolder(save_path_stft)
                mkdir(save_path_stft)
            end
            save_path_cwd = [result_path sub_str{sub} '\CWD_' num2str(num_class) 'cl\' num2str(label(trial)) '\'];
            if ~isfolder(save_path_cwd)
                mkdir(save_path_cwd)
            end
            
            image_ersp = mat2gray(tfdata_ersp(trial,:,:));
            image_stft = mat2gray(mag2db(abs(tfdata_stft(trial,:,:))));
            image_cwd = mat2gray(mag2db(abs(tfdata_cwd(trial,:,:))));
            
            imwrite(squeeze(image_ersp),[save_path_ersp num2str(trial) '.png'])
            imwrite(squeeze(image_stft),[save_path_stft num2str(trial) '.png'])
            imwrite(squeeze(image_cwd),[save_path_cwd num2str(trial) '.png'])

%             save([result_path sub_str{sub} '\CWD\' num2str(trial) ],'cwd_total_trial');
        end
        
        
        save([result_path sub_str{sub} '_ERSP'],'tfdata_ersp');
        save([result_path sub_str{sub} '_STFT'],'tfdata_stft');
        save([result_path sub_str{sub} '_CWD'],'tfdata_cwd','-v7.3');
        
        load([base_path sub_str{sub} '\vs_data_balanced']);
        label = balanced_vs_cate_label;
        label_9cl = balanced_vs_label -1;
        
        save([result_path sub_str{sub} '_ERSP'],'label', '-append');
        save([result_path sub_str{sub} '_ERSP'],'label_9cl', '-append');
        save([result_path sub_str{sub} '_STFT'],'label', '-append');
        save([result_path sub_str{sub} '_STFT'],'label_9cl', '-append');
        save([result_path sub_str{sub} '_CWD'],'label', '-append');
        save([result_path sub_str{sub} '_CWD'],'label_9cl', '-append');
        toc;
        
    end
    
    
end

disp('done');
disp('done');
%%
clear all
sub_str = {'S01','S02','S09','S18','S24','S28','S29'};
task_list = {'VS','VI', 'AS','AI'};


base_path = 'D:\EEG_data_Visual\VI\balanced\';

num_class = 9;
addpath('D:\Envision_EEG_TFdata\2D\eeglab2022.0');
result_path = 'D:\EEG_TFMap\';
if ~isfolder(result_path)
    mkdir(result_path)
end

for task = 2:2
    
    for sub =1: length(sub_str)
        disp([sub_str{sub} 'started']);
        tic
        % trial x channel x time
        
        load([base_path sub_str{sub} '\vi_data_balanced']);
        label = balanced_vi_cate_label;
        label_9cl = balanced_vi_label -1;
        
        save([result_path sub_str{sub} '_ERSP'],'label', '-append');
        save([result_path sub_str{sub} '_ERSP'],'label_9cl', '-append');
        save([result_path sub_str{sub} '_STFT'],'label', '-append');
        save([result_path sub_str{sub} '_STFT'],'label_9cl', '-append');
        save([result_path sub_str{sub} '_CWD'],'label', '-append');
        save([result_path sub_str{sub} '_CWD'],'label_9cl', '-append');
        toc;
        
    end
    
    
end
%% plotting

figure();
x = linspace(0,4000,128);
y = linspace(0,64,1024);
imagesc(x,y,tf'); xlabel('Time (ms)', 'FontSize',14);
ylabel('Frequency (Hz)', 'FontSize',14)
set(gca,'FontSize',14);colorbar;
set(gca,'YDir','normal')
title([biosemilocch32.label(channel)]);
rectangle('Position',[0 0 4000 50], 'EdgeColor','r', 'LineWidth',2)
caxis([-5 5]);


figure();
y = linspace(1,384,384); x = linspace(0,4000,512);
imagesc(x,y,mat2gray(squeeze(tfdata_cwd)))
imagesc(x,y,squeeze(tfdata_cwd))
y = linspace(1,384,32);
yticks(y)
yticklabels(biosemilocch32.label)
set(gca,'FontSize',12);
set(gca,'YDir','normal')
ylabel('Channel','FontSize',16);xlabel('Time (ms)','FontSize',16);colorbar;
% colormap('gray');
colormap('default');
caxis([-5 5]);


 load('D:\EEG_data_Visual\CSP\VI_9class\S01\Fold1\csp_feature.mat')
figure(); topoplot(csp_feature{1,1}.train(26,:),'D:\biosemi32.loc','electrodes','ptslabels');caxis([8 10])




