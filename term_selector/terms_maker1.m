% autuanliu@163.com
% 2019年2月20日
% main entry
% 多 trial 版本
%

tic;
clear;

% 参数设置
% scale_type = 'mapminmax';     % !set{'mapminmax', 'zscore'}
scale_type = 'none';     % !set{'mapminmax', 'zscore'} 避免归一化
max_lag = 10;   % 不影响结果
threshold = 5;
n_trial = 100;
root = '../data/';

for m={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}
    flag = m{1, 1};             % !set{'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}
    for trial = 1:n_trial
        disp([flag, ' trial ### ', int2str(trial)]);
        [normalized_signals, Hv, Kalman_H, S_No, ERRs, terms_chosen] = make_terms(flag, max_lag, scale_type, threshold, '_signals5D_noise100.mat', trial);
        % 保存重要数据
        disp('saving important data ......');
        name_set = {['linear_terms', int2str(trial), '.mat'], ['nonlinear_terms', int2str(trial), '.mat'], ['longlag_linear_terms', int2str(trial), '.mat'], ['longlag_nonlinear_terms', int2str(trial), '.mat']};

        switch flag
            case 'linear'
                save([root, name_set{1, 1}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
            case 'nonlinear'
                save([root, name_set{1, 2}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
            case 'longlag_linear'
                save([root, name_set{1, 3}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
            case 'longlag_nonlinear'
                save([root, name_set{1, 4}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
            otherwise
                disp('Not Define!')
        end
    end
end
toc;

function [normalized_signals, Hv, Kalman_H, S_No, ERRs, terms_chosen] = make_terms(flag, max_lag, scale_type, threshold, postfix, trial)
    format long;
    root = '../data/';

    % 读取数据
    switch flag
        case 'linear'
            norder = 1;
            len1 = 25;
            max_lag = 5;    % 最大时延
            load([root, flag, postfix]);   % linear signals
            signals = eval([flag, '_signals100']);
            signals = squeeze(signals(trial, :, :));
        case 'nonlinear'
            max_lag = 5;    % 最大时延
            norder = 2;
            len1 = 350;
            load([root, flag, postfix]);   % nonlinear signals
            signals = eval([flag, '_signals100']);
            signals = squeeze(signals(trial, :, :));
        case 'longlag_linear'
            max_lag = 10;    % 最大时延
            norder = 1;
            len1 = 50;
            load([root, flag, postfix]);   % longlag linear signals
            signals = eval([flag, '_signals100']);
            signals = squeeze(signals(trial, :, :));
        case 'longlag_nonlinear'
            max_lag = 10;    % 最大时延
            norder = 2;
            len1 = 1325;
            load([root, flag, postfix]);   % longlag nonlinear signals
            signals = eval([flag, '_signals100']);
            signals = squeeze(signals(trial, :, :));
        otherwise
            disp('Not Define!')
    end


    %% !数据标准化处理
    % ! scaler_type set = {'mapminmax', 'zscore'}
    switch scale_type
        case 'mapminmax'
            normalized_signals = normalize(signals, 'mapminmax');
        case 'zscore'
            normalized_signals = normalize(signals, 'zscore');
        otherwise
            normalized_signals = signals;
    end
    %%

    % % 数据存储
    % normalized_signals100 = zeros(100, 2048, 5);
    % Kalman_H100 = zeros(100, 5, 2048-max_lag, 5);
    % S_No100 = zeros(100, 5, len1);
    % ERRs100 = zeros(100, 5, 5);
    % terms_chosen100 = zeros(100, 5, 5);

    %% !基于RFOLS 算法的候选项选择器
    [H, Hv] = buildH(normalized_signals, norder, max_lag);
    [Kalman_H, ~, ~, S_No, ERRs] = term_selector(normalized_signals, norder, max_lag, H, threshold);
    terms_chosen = S_No(:, 1:threshold);   % threshold 为候选项的个数
    % normalized_signals100(trial, :, :) = normalized_signals;
    % Kalman_H100(trial, :, :, :) = Kalman_H;
    % ERRs100(trial, :, :) = ERRs;
    % terms_chosen100(trial, :, :) = terms_chosen;
end
