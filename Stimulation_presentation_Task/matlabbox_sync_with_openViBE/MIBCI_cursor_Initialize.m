% MIBCI_cursor_Initialize.m
% -------------------------------
% Author : Kyungho Won (GIST)
% Date   : 8 June 2021
%

function box_out = MIBCI_cursor_Initialize(box_in)
	
    % we display the setting values
    disp('Box settings are:')
    
    % load the trained CSP and LDA mdl
    mdl_path = box_in.settings(1).value;
    train_mdl = load(mdl_path);
    disp(train_mdl);
    
    % ch and other parameter settings are based on DSI-24 device
    if strcmpi(box_in.settings(2).value, 'on')
        box_in.user_data.chCSP = 1:19;
    elseif strcmpi(box_in.settings(2).value, 'off')
        box_in.user_data.chCSP = 1:9;
    end
    disp(box_in.user_data.chCSP);
    
    box_in.user_data.bandpass = [8 30];
    box_in.user_data.bandstop = [58 60];
    box_in.user_data.nbfilter = box_in.settings(3).value; % CSP filters
    box_in.user_data.srate = 300;
    box_in.user_data.nCh = 19; % whole ch. except for extra
    
    % send the model to Process() 
    box_in.user_data.train_mdl = train_mdl;
    % train_mdl
    %   - train_mdl.CSP -> .W, .D
    %   - train_mdl.mdl -> lda_W, lda_w0
    %   - train_mdl.mld -> svm_mdl
    
	% let's add a user-defined indicator to know if the output header is set
    box_in.user_data.is_headerset = false;
	
	% We also add some statistics
	box_in.user_data.nb_matrix_processed = 0;
	box_in.user_data.mean_fft_matrix = 0;
    
    % for bandpass filtering
    ft_defaults;
	
    box_out = box_in;
   
end
    