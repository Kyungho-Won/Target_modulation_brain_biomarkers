% Main MI Practice.m - currently used
% Parameters: nb trial (left + right hand MI)
%             t_cross, t_instruction, t_feedback, t_blank (ITI)
clc; clear;
addpath('./functions');
addpath('./images_MI');

% load images for MI practice
imgs_task{1} = imread('./images_MI/cross.JPG');
imgs_task{2} = imread('./images_MI/left.JPG');
imgs_task{3} = imread('./images_MI/right.JPG');
imgs_task{4} = imread('./images_MI/blank.JPG');

% default parameters
Params = struct('nbtrial', 50, 't_cross', 1.5, ...
    't_instruction', 4.0, 't_feedback', 2.0, 't_blank', 1.5, 'online', 'on');
rnd_trial = [2*ones(1,floor(Params.nbtrial/2)) 3*ones(1,floor(Params.nbtrial/2))];
rnd_trial = shuffle(rnd_trial);

toOV = tcpip('localhost',15361); % to OV
fclose(toOV); % close if there are previous session remained
fopen(toOV);

fromOV = tcpclient('localhost', 5678); % From OV

figure('WindowState', 'fullscreen', 'MenuBar', 'none', 'ToolBar', 'none');
image(imgs_task{4});
img_handle = get(gca, 'Children');
axis off;
set(gcf, 'Color', 'k');

send_to_OV(toOV, 3);
pause(4.0);
header_OV = read(fromOV); % drop header from OV
w = waitforbuttonpress;

for nTrial=1:Params.nbtrial
    tic_trial = tic;
    % cross
    draw_cross(img_handle, imgs_task{1});
    pause(Params.t_cross);
    
    % instruction (arrow)
    send_to_OV(toOV, 3);
    send_to_OV(toOV, rnd_trial(nTrial)-1); % 1: left, 2: right
    draw_instruction(img_handle, imgs_task{rnd_trial(nTrial)});
    pause(Params.t_instruction);
    
    % visual feedback
    if strcmpi(Params.online, 'on')
        OV_dat = read(fromOV);
        OV_dat_cast = typecast(OV_dat, 'double');
        disp(OV_dat_cast);
        tic_draw_plt = tic;
        draw_feedback(OV_dat_cast);
        pause(Params.t_feedback-toc(tic_draw_plt));
    end

    % blank
    begin_draw_blank = tic;
    draw_blank(imgs_task{4});
    end_draw_blank = toc(begin_draw_blank);
    img_handle = get(gca, 'Children');
    pause(Params.t_blank-end_draw_blank);
    
    elapsed_trial = toc(tic_trial);    
    disp(elapsed_trial);
end

close all;
clear toOV fromOV
fclose('all');
echotcpip('off');

%% sub functions

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


