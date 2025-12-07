%%% cleaning data with eeglab for encoding decoding 
%% initialization
clc; clear; close all;path(pathdef);matlabrc
path_toolbox = 'C:\Dev\MATLAB_Toolboxes\MatlabPreprocessing\';
addpath(strcat(path_toolbox,'eeglab2024.0\'))
eeglab;
addpath(strcat(path_toolbox,'eeglab2024.0\plugins\ICLabel\viewprops'))%topoplot ICA component
dataPath = 'C:\Users\yanni\Desktop\TUM MSEI\2. Semester\Lab\Sub01';
vhdrName1 = 'Yanick.vhdr';
close all
%%
%%%%%%% Preprocessing steps %%%%%%%
%% Step1: import data
[EEG_raw, com11] = pop_loadbv(dataPath, vhdrName1); %loading data
EEG_raw = pop_chanedit(EEG_raw); %import channel

EEG_raw.event(133).type = 'S 23';
EEG_raw.urevent(133).type = 'S 23';
%% Step2: filter data
%spectopo(EEG_raw.data, 0, EEG_raw.srate,'title','spectra','limits',[0 60 NaN NaN NaN NaN]);
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
%spectopo(EEG_raw.data, 0, EEG_raw.srate,'title','spectra','limits',[0 60 NaN NaN NaN NaN]);
%pop_spectopo(EEG_raw, 1); %plot channels
%% Step3: resampling
[EEG_raw] = pop_resample(EEG_raw, 500); %resample to 500 Hz
%% Step4: epoching
EEG_raw = pop_epoch(EEG_raw, {'S  2'}, [-3 ,32]);
pop_eegplot(EEG_raw) %% Check which channels and trials need to be interpolated
%% step5: interpolation bad channels in some trials
EEG_raw = eeg_interp(EEG_raw, 21, 'spherical',[1:4 6:60]);%% NO interpolation 
EEG_raw = eeg_interp(EEG_raw, 24, 'spherical',[1:56 58:60]);%% NO interpolation 
EEG_raw = eeg_interp(EEG_raw, 22, 'spherical',[1:48 50:60]);%% NO interpolation 

EEG_raw = pop_select(EEG_raw,'notrial',[42 49 59]);
%spectopo(EEG_raw.data, 0, EEG_raw.srate,'title','spectra','limits',[0 60 NaN NaN NaN NaN]);
%% step6 part1: ICA
EEG_ica = pop_runica(EEG_raw,'icatype','sobi');
EEG_ica = iclabel(EEG_ica);
pop_viewprops(EEG_ica,0,1:1:30);
%% step6 part2: ICA
data_eeg = pop_subcomp( EEG_ica, [1 3:5 8 10 11 12 14 15 16 17 19:21 23:27 29 30], 0);
pop_eegplot(data_eeg) %% Check which channels and trials need to be interpolated
%% remove EOG channels
%data_eeg = pop_select(data_eeg, 'nochannel', [15, 17, 47]);% removing 'EOG' channels
%% double check with fieldrip
eeg_data = eeglab2fieldtrip( data_eeg, 'raw', 'none' );
addpath 'C:\Users\simon\Documents\master\25WS\praktikum_eeg\code\eeg-classification\toolboxes\fieldtrip-20250106'
ft_defaults
cfg       	= [];
cfg.method	= 'summary';
eeg_data 	= ft_rejectvisual(cfg,eeg_data);
pop_viewprops(data_eeg,0,1:1:30);
%% save data
save('sub1_EnDe.mat','data_eeg','-v7.3')