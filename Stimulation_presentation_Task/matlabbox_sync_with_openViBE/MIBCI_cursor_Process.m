% tuto2_FFT_filter_Process.m
% -------------------------------
% Author : Laurent Bonnet (INRIA)
% Date   : 25 May 2012
%

function box_out = MIBCI_cursor_Process(box_in)

for i = 1: OV_getNbPendingInputChunk(box_in,1)
    
    if(~box_in.user_data.is_headerset)
        % The output is the input + noise, only on first channel
        box_in.outputs{1}.header = box_in.inputs{1}.header;
        box_in.outputs{1}.header.nb_channels = 19;
        box_in.user_data.is_headerset = 1;
        % We print the header in the console
        disp('Input header is :')
        box_in.inputs{1}.header
        disp('Output header is :')
        box_in.outputs{1}.header
    end
    
    
    [box_in, start_time, end_time, matrix_data] = OV_popInputBuffer(box_in,1);
    
    srate = box_in.user_data.srate;
    bp = box_in.user_data.bandpass;
    bs = box_in.user_data.bandstop;
    
    % filtering
    bp_dat = ft_preproc_bandpassfilter(matrix_data, srate, bp, 4, 'but');
    bp_dat = ft_preproc_bandstopfilter(bp_dat(box_in.user_data.chCSP, :), srate, bs, 4, 'but');
    
    % feature extraction
    train_mdl = box_in.user_data.train_mdl;
    nCh = length(box_in.user_data.chCSP);
    nbfilter = box_in.user_data.nbfilter;
    
    csp_pat = [1:nbfilter, nCh-nbfilter+1:nCh];
    cur_feat = log(var(train_mdl.CSP.W(:, csp_pat)'*bp_dat, [], 2));
    lda_predict = train_mdl.mdl.lda_W'*cur_feat+train_mdl.mdl.lda_w0;
    lda_predict = 2 * ((lda_predict>0)-0.5); % 1 for left, -1 for right
    
    if lda_predict==1
        disp('Classified: left');
    elseif lda_predict==-1
        disp('Classified: right');
    end
    
    sig = ones(size(matrix_data)) * lda_predict; % 1 for left, -1 for right
    box_in = OV_addOutputBuffer(box_in,1,start_time,end_time,sig);
end

box_out = box_in;
end

