% Main MI Cursor task.m - currently used
% Parameters: nb trial (left + right hand MI)
%             t_cross, t_instruction, t_feedback, t_blank (ITI)
clc; clear;
addpath('./functions');
addpath('./images_MI');

% load images for MI practice

% imgs_task = imags_MIBCI
imgs_task = load_imgs_task(); 

% default parameters
Params = struct('nbtrial', 50, 't_cross', 1.5, ...
    't_instruction', 4.0, 't_feedback', 2.0, 't_blank', 1.5, 'online', 'on');
rnd_trial = [2*ones(1,floor(Params.nbtrial/2)) 3*ones(1,floor(Params.nbtrial/2))];
rnd_trial = shuffle(rnd_trial);

toOV = tcpip('localhost',15361); % to OV
fclose(toOV); % close if there are previous session remained
fopen(toOV);

figure('WindowState', 'fullscreen', 'MenuBar', 'none', 'ToolBar', 'none');
image(imgs_task{8});
img_handle = get(gca, 'Children');
axis off;
set(gcf, 'Color', 'k');

send_to_OV(toOV, 3);
pause(4.0);
if strcmpi(Params.online, 'on')
    fromOV = tcpclient('localhost', 5678); % From OV
    header_OV = read(fromOV); % drop header from OV
end
w = waitforbuttonpress;

for nTrial=1:Params.nbtrial
    tic_trial = tic;
    % cross
    draw_cross(img_handle, imgs_task{1});
    pause(Params.t_cross);
    
    % instruction (arrow)
    send_to_OV(toOV, 3);
    current_instruction = rnd_trial(nTrial);
    send_to_OV(toOV, current_instruction-1); % 1: left, 2: right
    draw_instruction(img_handle, imgs_task{current_instruction});
    pause(Params.t_instruction);
    
    % visual feedback
    OV_dat = [];
    if strcmpi(Params.online, 'on')
        while isempty(OV_dat) % wait until packet is delivered
            OV_dat = read(fromOV);
        end
        OV_dat_cast = typecast(OV_dat, 'double');
        tic_draw_plt = tic;
        draw_feedback(img_handle, imgs_task, current_instruction-1, OV_dat_cast);
        send_to_OV(toOV, OV_dat_cast(end)+2+10);
        pause(Params.t_feedback-toc(tic_draw_plt));
    end

    % blank
    begin_draw_blank = tic;
    draw_blank(imgs_task{8});
    end_draw_blank = toc(begin_draw_blank);
    img_handle = get(gca, 'Children'); 
    % set(img_handle, 'CData', img) is faster than drawing a new image
    pause(Params.t_blank-end_draw_blank);
    
    elapsed_trial = toc(tic_trial);    
    disp(elapsed_trial);
end

close all;
clear toOV fromOV
fclose('all');
echotcpip('off');

%% sub functions

function out = load_imgs_task()
imgs_task{1} = imread('./images_MI/cross.JPG');
imgs_task{2} = imread('./images_MI/left_cursor.JPG');
imgs_task{3} = imread('./images_MI/right_cursor.JPG');
imgs_task{4} = imread('./images_MI/left_hit.JPG');
imgs_task{5} = imread('./images_MI/left_miss.JPG');
imgs_task{6} = imread('./images_MI/right_hit.JPG');
imgs_task{7} = imread('./images_MI/right_miss.JPG');
imgs_task{8} = imread('./images_MI/blank.JPG');

out = imgs_task;
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

function draw_feedback(img_handle, img, instruction, tmp_dat)

% tmp_dat(1): 1: left, -1: right
disp(tmp_dat(end));
if instruction == 1 % left
    % hit or miss
    if tmp_dat(end) > 0
        set(img_handle, 'CData', img{4}); % hit
        disp('hit');
    else
        set(img_handle, 'CData', img{5}); % miss
        disp('miss');
    end
elseif instruction == 2 % right
    % hit or miss
    if tmp_dat(end) < 0
        set(img_handle, 'CData', img{6}); % hit
        disp('hit');
    else
        set(img_handle, 'CData', img{7}); % miss
        disp('miss');
    end
end

end

function draw_blank(img)
image(img);
axis off;
set(gcf, 'Color', 'k');
end
