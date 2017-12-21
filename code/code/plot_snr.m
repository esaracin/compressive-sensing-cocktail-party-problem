clearvars
close all
clc
fids = {'../data/mixes/eastwood_graham_mix.wav', 'output_0.wav', 'output_1.wav', '../data/groundtruth/eastwood_lawyers.wav', '../data/groundtruth/graham_any_nation.wav'};
names = {'Mixed Data', 'Recovered Source 1', 'Recovered Source 2', 'Groundtruth 1', 'Groundtruth 2'};

for i = 1:length(fids)
    figure()
    [signal, Fs] = audioread(fids{i});
    fprintf('%s : Sampled %d hz\n', fids{i}, Fs)
    N = length(signal);
    signalToNoise = snr(signal, Fs);
    fprintf('\tSNR for %s : %f\n', fids{i}, signalToNoise)
    X_mags = abs(fft(signal));
    fax_bins = [0 : N-1];
    N_2 = ceil(N/2);
    plot(fax_bins(1:N_2), X_mags(1:N_2))
    xlabel('Frequency (Bins)')
    ylabel('Magnitude');
    title({'Single-sided Magnitude Spectrum'; sprintf('%s', names{i})});
    axis tight
    saveas(gcf, sprintf('../../paper/figs/%s.png', names{i}))
end
