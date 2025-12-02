%%% cleaning data with eeglab for encoding decoding 
%% initialization
clc; clear; close all;path(pathdef);matlabrc
addpath('F:\EEGdata\Toolbox\eeglab2021.1')
eeglab;
addpath('F:\EEGdata\Toolbox\eeglab2021.1\plugins\ICLabel\viewprops')%topoplot ICA component
dataPath = 'H:\PhD\PhD_data\03_SpeechFamiliarization\2021.11.19.JuanDaniel\';
vhdrName1 = 'JuanDaniel.vhdr';
vhdrName2 = 'JuanDaniel2.vhdr';
%%
%%%%%%% Preprocessing steps %%%%%%%
%% Step1: import data
[EEG_raw1, com11] = pop_loadbv(dataPath, vhdrName1); %loading data
EEG_raw1 = pop_chanedit(EEG_raw1); %import channel
EEG_raw1 = pop_select(EEG_raw1, 'nochannel', [63, 64]);% removing 'HR' and 'GSR' channels
[EEG_raw2, com12] = pop_loadbv(dataPath, vhdrName2); %loading data
EEG_raw2 = pop_chanedit(EEG_raw2); %import channel
EEG_raw2 = pop_select(EEG_raw2, 'nochannel', [63, 64]);% removing 'HR' and 'GSR' channels

%%%% concatinate two files
EEG_raw = pop_mergeset( EEG_raw1, EEG_raw2, 0);
clear EEG_raw1 EEG_raw2
%%% correction of conditions
EEG_raw.event(109).type = 'S  3';
EEG_raw.urevent(109).type = 'S  3';
EEG_raw.event(113).type = 'S 22';
EEG_raw.urevent(113).type = 'S 22';
EEG_raw.event(119).type = 'S 23';
EEG_raw.urevent(119).type = 'S 23';
%% Step2: filter data
spectopo(EEG_raw.data, 0, EEG_raw.srate,'title','spectra','limits',[0 60 NaN NaN NaN NaN]);
[EEG_raw, com2, b1] = pop_eegfiltnew(EEG_raw, 0.5, 60, [], 0, 0, 1); %filter data [0.5 60]
%%%%%%%%%%%%%%%%%%%%%%% Explaination for paper
%%%%%% performing 6601 point bandpass filtering.
%%%%%% transition band width: 0.5 Hz
%%%%%% passband edge(s): [0.5 60] Hz
%%%%%% cutoff frequency(ies) (-6 dB): [0.25 60.25] Hz
%%%%%% filtering the data (zero-phase, non-causal)
%%%%%%%%%%%%%%%%%%%%%%% Explaination for paper
[EEG_raw, com3, b2] = pop_eegfiltnew(EEG_raw, 48, 52, [], 1, 0, 1); %filter data notch [48 52]
%%%%%% performing 1651 point bandstop (notch) filtering.
%%%%%% transition band width: 2 Hz
%%%%%% passband edge(s): [48 52] Hz
%%%%%% cutoff frequency(ies) (-6 dB): [49 51] Hz
%%%%%% filtering the data (zero-phase, non-causal)
spectopo(EEG_raw.data, 0, EEG_raw.srate,'title','spectra','limits',[0 60 NaN NaN NaN NaN]);
%pop_spectopo(EEG_raw, 1); %plot channels
%% Step3: resampling
[EEG_raw] = pop_resample(EEG_raw, 500); %resample to 500 Hz
%% Step4: epoching
EEG_raw = pop_epoch(EEG_raw, {'S  1', 'S  2', 'S  3'}, [-3 ,65]);
pop_eegplot(EEG_raw) %% Check which channels and trials need to be interpolated
%% step5: interpolation bad channels in some trials
EEG_raw = eeg_interp(EEG_raw, 18, 'spherical',1:1:66);%% NO interpolation 
spectopo(EEG_raw.data, 0, EEG_raw.srate,'title','spectra','limits',[0 60 NaN NaN NaN NaN]);
%% step6 part1: ICA
EEG_ica = pop_runica(EEG_raw,'icatype','sobi');
EEG_ica = iclabel(EEG_ica);
pop_viewprops(EEG_ica,0,1:1:18);
%% step6 part2: ICA
data_eeg = pop_subcomp( EEG_ica, [1 3 4 10 11 13 16:1:18 20:1:22 24 26:1:31 33:1:35 37 38 40 41 43 44 47 48 51:1:54 56:1:62], 0);
pop_eegplot(data_eeg) %% Check which channels and trials need to be interpolated
%% remove EOG channels
data_eeg = pop_select(data_eeg, 'nochannel', [15, 17, 47]);% removing 'EOG' channels
%% double check with fieldrip
eeg_data = eeglab2fieldtrip( data_eeg, 'raw', 'none' );
addpath 'F:\EEGdata\SpeechFamiliarization\fieldtrip-20200607'
ft_defaults
cfg       	= [];
cfg.method	= 'summary';
eeg_data 	= ft_rejectvisual(cfg,eeg_data);
pop_viewprops(data_eeg,0,1:1:24);
%% save data
save('sub1_EnDe.mat','data_eeg','-v7.3')