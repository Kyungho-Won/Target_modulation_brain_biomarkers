% Main MI Practice.m
% Parameters: nb trial (left + right hand MI)
%             t_cross, t_instruction, t_feedback, t_blank (ITI)
clc; clear;

% DSI-streamer channel locatino
ch_locs_streamer = {'FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', ...
    'C4', 'T3', 'T4', 'T5', 'T6', 'Pz','P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'Trigger'};

% load images for MI practice
fs = 300;
cd('C:/Users/BioComput_HGKim/Desktop/khwon/Stim_for_MI');
addpath('./functions');
addpath('./images_MI');
imgs_task = load_imgs_task();

% Initialize the TCP/IP with OpenViBE
t = tcpip('localhost',8844);
fclose(t); %just in case it was open from a previous iteration
fopen(t); %opens the TCPIP connection

% Initialize the serial with Arduino
fp_serial = serialport("COM8",9600);
configureTerminator(fp_serial, "CR/LF");
flush(fp_serial); % to remove any old data
% write(a, 1:5, "unit8"); % write [1 2 3 4 5] in unit8 format
% read(a, len, "unit8"); % read the data as much as len

time_vec = floor(clock);
fname_phase = sprintf('Phase_%d-%d-%d-%d-%d.mat', time_vec(2), time_vec(3), time_vec(4), ...
    time_vec(5), time_vec(6));

whole_phase = [];
stim_phase = [];

%% Main

cutoffcounter = 0;
% default parameters
Params = struct('nbtrial', 20, 't_cross', 1.5, ...
    't_instruction', 4.0, 't_feedback', 2.0, 't_blank', 1.5, 'online', 'off');
rnd_trial = [2*ones(1,floor(Params.nbtrial/2)) 3*ones(1,floor(Params.nbtrial/2))];
rnd_trial = shuffle(rnd_trial);
% stim parameters
len_window = 1000; % in ms
slides = 100; % in ms
params_stim = struct('freq', [8 13], 'frame', len_window);

squeezed_fig('WindowState', 'fullscreen', 'MenuBar', 'none', 'ToolBar', 'none');
image(imgs_task{8});
img_handle = get(gca, 'Children');
axis off;
set(gcf, 'Color', 'k');

pause(3.0);
w = waitforbuttonpress;
tic_Running = [];

for nTrial=1:Params.nbtrial
    tic_trial = tic;
    % s1. cross
    draw_cross(img_handle, imgs_task{1});
    pause(Params.t_cross);
    
    % s2-1. instruction (arrow)
    current_class = rnd_trial(nTrial);
    draw_instruction(img_handle, imgs_task{current_class});
    drawnow;
    
    % s2-2. collect epochs for phase tracking
    timeLog = [];
    signalLog = [];
    isRunning = false; % motor is already running
    tic;
    stim_tic = tic;
    while 1
        
        % extarct epoch: length, slides
        [timeLog, signalLog] = read_dat_DSI(t, timeLog, signalLog, floor(len_window/1000*fs));
        disp(size(signalLog)); % signal log: [ch x time] now 25 x data (24+event)
        
        % activate vibrotactile stimulation
        
        % current_class: 2 (left), 3 (right)
        % EEG: 2 (C3), 6 (C4)
        switch current_class % contralateral signal
            case 2 % left
                eeg = signalLog(10, :); % C4
            case 3 % right
                eeg = signalLog(9, :); % C3
        end
        cur_phase = tactile_stim_on_phase(eeg, fs, params_stim, 'off');
        whole_phase = cat(1, whole_phase, cur_phase);
        if (~isRunning)
            % falling phase
            if cur_phase >= -1 && cur_phase <= 0
                write(fp_serial, current_class-1+'0', "char"); % write 1 or 2 (left or right)
                stim_phase = cat(1, stim_phase, cur_phase);
            end
            
            isRunning = true;
            tic_Running = tic;
        end
        
        if ~isempty(tic_Running) && toc(tic_Running) >= 0.15
            isRunning = false;
        end
        
        % slide buffer
        signalLog(:, 1:floor(slides/1000*fs)) = [];
        timeLog(1:floor(slides/1000*fs)) = [];
        %         disp(size(signalLog));
        
        % timer - only run this loop for t_instruction
        if toc(stim_tic) >= Params.t_instruction
            break;
        end
    end
    toc;
    % ---------------------------------------------- instruction ends ----
    
    % s3. blank
    begin_draw_blank = tic;
    draw_blank(imgs_task{8});
    end_draw_blank = toc(begin_draw_blank);
    img_handle = get(gca, 'Children');
    pause(Params.t_blank-end_draw_blank);
    
    elapsed_trial = toc(tic_trial);
    disp(elapsed_trial);
end

save(fname_phase, 'whole_phase', 'stim_phase');
close all;
clear t fp_serial;
fclose('all');
echotcpip('off');

%% sub functions

function [timeLog, signalLog] = read_dat_DSI(t, timeLog, buffer, buffer_size)
headerStart = [64, 65, 66, 67, 68]; % All DSI packet headers begin with '@ABCD', corresponding to these ASCII codes.
cutoffcounter = 0;
notDone = 1;

% until buffer is full
while notDone
    
    % Read the packet
    data = uint8(fread(t, 12))'; % Loads the first 12 bytes of the first packet, which should be the header
    data = [data, uint8(fread(t, double(typecast(fliplr(data(7:8)), 'uint16'))))']; % Loads the full packet, based on the header
    lengthdata = length(data);
    
    if all(ismember(headerStart,data)) % Checks if the packet contains the header
        packetType = data(6); %this determines whether it's an event or sensor packet.
        
        % Event Packet.  This includes the greeting packet
        if packetType == 5
            nodeId = typecast(fliplr(data(16:19)), 'uint32');
            if ((typecast(fliplr(data(13:16)), 'uint32') ~= 2) &&  (typecast(fliplr(data(13:16)), 'uint32') ~=3))
                messagelength = typecast(fliplr(data(21:24)), 'uint32');
                message = char(data(25:24+messagelength));
            elseif (typecast(fliplr(data(13:16)), 'uint32')) == 2
                disp('Here is the start of the data.')
            else
                disp('The data has ended.')
            end
        end
        
        % EEG sensor packet
        if packetType == 1
            Timestamp = swapbytes(typecast(data(13:16),'single'));
            EEGdata = swapbytes(typecast(data(24:lengthdata),'single'));
            
            timeLog = [timeLog, Timestamp];
            buffer = [buffer, EEGdata'];
            
            % Limit the size of the logs to buffer_size data points
            if length(timeLog) >= buffer_size
                signalLog = buffer;
                break;
            end
            
        end
        
        % Termination clause
        if t.Bytesavailable < 12                %if there's not even enough data available to read the header
            cutoffcounter = cutoffcounter + 1;  %take a step towards terminating the whole thing
            if cutoffcounter == 3000            %and if 3000 steps go by without any new data,
                notDone = 0;                     %terminate the loop.
            end
            disp('no bytes available')
            pause(0.001)
        else  %meaning, unless there's data available.
            cutoffcounter = 0;
            
        end
        
    end
    
end


end

function draw_cross(img_handle, img)
set(img_handle, 'CData', img);
% axis off;
% set(gcf, 'Color', 'k');
end

function draw_instruction(img_handle, img)
set(img_handle, 'CData', img);
% axis off;
% set(gcf, 'Color', 'k');
end

function draw_feedback(tmp_dat)
bh = bar([tmp_dat(1) tmp_dat(2)], 0.5, 'facecolor', 'flat');
bh.CData(1,:) = [0 0 1];
bh.CData(2,:) = [1 0 0];
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');

set(gca, 'linewidth', 2);
set(gcf, 'Color', 'k');
% xlim([-max(get(gca, 'Xlim')) max(get(gca, 'Xlim'))]);
axis off;
end

function draw_blank(img)
image(img);
axis off;
set(gcf, 'Color', 'k');
end

function out = load_imgs_task()
imgs_task{1} = imread('./images_MI/cross.JPG');
imgs_task{2} = imread('./images_MI/left_cursor_wh.JPG');
imgs_task{3} = imread('./images_MI/right_cursor_wh.JPG');
imgs_task{4} = imread('./images_MI/left_hit.JPG');
imgs_task{5} = imread('./images_MI/left_miss.JPG');
imgs_task{6} = imread('./images_MI/right_hit.JPG');
imgs_task{7} = imread('./images_MI/right_miss.JPG');
imgs_task{8} = imread('./images_MI/blank.JPG');

out = imgs_task;
end


