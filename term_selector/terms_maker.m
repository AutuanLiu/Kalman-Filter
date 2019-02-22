% autuanliu@163.com
% 2018年12月10日
% main entry
%

tic;
clear;

% 参数设置
scale_type = 'mapminmax';     % !set{'mapminmax', 'zscore'}
max_lag = 10;
threshold = 5;

for t=[true, false]
    for m={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}
        is_normalize = t;           % ! 是否标准化数据
        flag = m{1, 1};             % !set{'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}
        make_terms(flag, is_normalize, max_lag, scale_type, threshold);
    end
end
toc;

function [] = make_terms(flag, is_normalize, max_lag, scale_type, threshold)
    format long;
    root = '../data/';

    % 读取数据
    switch flag
        case 'linear'
            norder = 1;
            max_lag = 5;    % 最大时延
            load([root, flag, '_signals5D_noise1.mat']);   % linear signals
            signals = eval([flag, '_signals']);
        case 'nonlinear'
            max_lag = 5;    % 最大时延
            norder = 2;
            load([root, flag, '_signals5D_noise1.mat']);   % nonlinear signals
            signals = eval([flag, '_signals']);
        case 'longlag_linear'
            max_lag = 10;    % 最大时延
            norder = 1;
            load([root, flag, '_signals5D_noise1.mat']);   % longlag linear signals
            signals = eval([flag, '_signals']);
        case 'longlag_nonlinear'
            max_lag = 10;    % 最大时延
            norder = 2;
            load([root, flag, '_signals5D_noise1.mat']);   % longlag nonlinear signals
            signals = eval([flag, '_signals']);
        otherwise
            disp('Not Define!')
    end


    %% !数据标准化处理
    % ! scaler_type set = {'mapminmax', 'zscore'}
    if is_normalize
        switch scale_type
            case 'mapminmax'
                normalized_signals = normalize(signals, 'mapminmax');
            case 'zscore'
                normalized_signals = normalize(signals, 'zscore');
            otherwise
                disp('Not Define!')
        end
    else
        normalized_signals = signals;
    end
    %%

    %% !基于RFOLS 算法的候选项选择器
    [H, Hv] = buildH(normalized_signals, norder, max_lag, 0);
    [Kalman_H, sparse_H, S, S_No, ERRs] = term_selector(normalized_signals, norder, max_lag, H, threshold);
    terms_chosen = S_No(:, 1:threshold);   % threshold 为候选项的个数

    % 保存重要数据
    disp('saving important data ......');
    if is_normalize
        name_set = {'nor_linear_terms.mat', 'nor_nonlinear_terms.mat', 'nor_longlag_linear_terms.mat', 'nor_longlag_nonlinear_terms.mat'};
    else
        name_set = {'linear_terms.mat', 'nonlinear_terms.mat', 'longlag_linear_terms.mat', 'longlag_nonlinear_terms.mat'};
    end

    switch flag
        case 'linear'
            save([root, name_set{1, 1}], 'normalized_signals', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'ERRs', 'terms_chosen');  % linear signals
        case 'nonlinear'
            save([root, name_set{1, 2}], 'normalized_signals', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'ERRs', 'terms_chosen');  % nonlinear signals
        case 'longlag_linear'
            save([root, name_set{1, 3}], 'normalized_signals', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'ERRs', 'terms_chosen');  % longlag linear signals
        case 'longlag_nonlinear'
            save([root, name_set{1, 4}], 'normalized_signals', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'ERRs', 'terms_chosen');  % longlag nonlinear signals
        otherwise
            disp('Not Define!')
    end
end
