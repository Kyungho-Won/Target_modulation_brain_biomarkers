function out = tactile_stim_on_phase(data, fs, params_stim, opt_plot)
%
% tactile_stim_on_phase(): send appropriate flag to serial device
%                          for VT (vibrotactile) stimulation
% inputs:
%    - fp: file pointer (serial device)
%    - data: segmented EEG [1 x time]
%    - fs: sampling frequency (Hz)
%    - class: current trial can be left/right [2 or 3 for left/right]
%    - params_stim (struct)
%     * frame: time in ms
%     * range: when to deliver stim
%     * freq: band pass filtering
%
%     - opt_plot: 'on': display plot otherwise no display
% output:
%    - predicted phase (for post analysis)

% data should be zero-mean, single channel
% data = ft_preproc_bandpassfilter(data, fs, params_stim.freq, 5, 'but');
data = iirfilt(data, fs, min(params_stim.freq), max(params_stim.freq));

% extact dominant frequency using FFT
t = 0:1/fs:params_stim.frame/1000-1/fs;
L = fs*2;
y = fft(data, L);
y = fftshift(y); % shift zero-frequency component to center of spectrum
f = (-L/2:L/2-1)/L*fs;

if strcmpi(opt_plot, 'on')
    figure,
    subplot(5,1,1);
    plot(t*1000, data); box off;
    xlabel('time (ms)'); ylabel('\muV');
    set(gca, 'fontsize', 13);
    
    subplot(5,1,2);
    plot(f, abs(y)); box off;
    xlabel('Frequency (Hz)');
    set(gca, 'fontsize', 13);
    
end

[~, max_fft] = max(abs(y));
dominant_f = f(max_fft); % find the dominant frequency

theta = angle(y); % phase anlge in radian

if strcmpi(opt_plot, 'on')
    subplot(5,1,3);
    stem(f, theta/pi); box off;
    ylabel('Phase / \pi');
    xlabel('Frequency (Hz)');
    
%     xline(dominant_f*[-1 1], 'color', 'r', 'linewidth', 2);
    set(gca, 'fontsize', 13);
end

% predict incoming EEG by fitting simple sine function
t_forecast = t(end)+1/fs:1/fs:2*t(end)+1/fs;
forecast_dat = cos(2*pi*abs(dominant_f)*t_forecast-theta(max_fft));

if strcmpi(opt_plot, 'on')
    subplot(5,1,4);
    plot(t*1000, (data-mean(data))/max(abs(data))); hold on;
    
    plot(t_forecast*1000, forecast_dat, 'color', 'r'); box off;
    set(gca, 'fontsize', 13);
    legend({'current', 'forecast'});
    xlabel('time (ms)');
    
end

% phase from the current window
yh_cur = hilbert(data);
phase_cur = angle(yh_cur);

% predict incoming phase
yh = hilbert(forecast_dat);
phase = angle(yh);

predicted_phase = phase(1)/pi; % in pi
% range: -1 to 1

out.phase = predicted_phase;
out.t_pred = t_forecast;
out.dat_pred = forecast_dat;
end