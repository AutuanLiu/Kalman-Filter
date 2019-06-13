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
% data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};
data_type = {'linear', 'nonlinear'};
max_lag = 10;      % 不影响结果
threshold = 5;
ndim = 5;          % 信号的通道数 {5, 10, 20}
n_trial = 100;
root = '../data/';

for err_var = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    for ndim=[5, 10]
        for m = data_type
            flag = m{1, 1};             % !set{'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}
            disp(['data_type: ', flag, '    ndim: ', int2str(ndim), '    err_var: ', sprintf('%2.2f', err_var)])
            for trial = 1:n_trial
                disp([flag, ' trial ### ', int2str(trial)]);
                % is_wgci 是否计算和 WGCI 相关的数据
                % for is_wgci = [0, 1]
                    % switch is_wgci
                % name_set = {['linear_terms', int2str(trial), '.mat'], ['nonlinear_terms', int2str(trial), '.mat'], ['longlag_linear_terms', int2str(trial), '.mat'], ['longlag_nonlinear_terms', int2str(trial), '.mat']};
                suffix = ['_signals', int2str(ndim), 'D_noise100_', sprintf('%2.2f', err_var), '.mat'];
                postfix = [int2str(ndim), 'D_', sprintf('%2.2f', err_var), 'trial', int2str(trial), '.mat'];
                name_set = {['linear_terms', postfix], ['nonlinear_terms', postfix]};
                [normalized_signals, Hv, Kalman_H, S_No, ERRs, terms_chosen] = make_terms(flag, max_lag, scale_type, threshold, suffix, trial, 0);
                        % case 0
                            % [normalized_signals, Hv, Kalman_H, S_No, ERRs, terms_chosen] = make_terms(flag, max_lag, scale_type, threshold, suffix, trial, 0);
                        % case 1
                        %     Kalman_H = cell(1, ndim);
                        %     ERRs = cell(1, ndim);
                        %     terms_chosen = cell(1, ndim);
                        %     for id = 1:ndim
                        %         [normalized_signals, Hv, Kalman_H1, S_No, ERRs1, terms_chosen1] = make_terms(flag, max_lag, scale_type, threshold, suffix, trial, id);
                        %         Kalman_H{1, id} = Kalman_H1;
                        %         ERRs{1, id} = ERRs1;
                        %         terms_chosen{1, id} = terms_chosen1;
                        %     end
                        % otherwise
                        %     disp('ERROR!')
                        % end
                    % end
                    
                % 保存重要数据
                % disp('saving important data with WGCI ......');
                disp('saving important data ......');
                switch flag
                    case 'linear'
                        save([root, name_set{1, 1}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
                    case 'nonlinear'
                        save([root, name_set{1, 2}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
                    % case 'longlag_linear'
                    %     save([root, name_set{1, 3}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
                    % case 'longlag_nonlinear'
                    %     save([root, name_set{1, 4}], 'normalized_signals', 'Hv', 'Kalman_H', 'ERRs', 'terms_chosen');
                    otherwise
                        disp('Not Define!')
                end
            end
        end
    end
end
toc;

function [normalized_signals, Hv, Kalman_H, S_No, ERRs, terms_chosen] = make_terms(flag, max_lag, scale_type, threshold, postfix, trial, id_wgci)
    format long;
    root = '../data/';

    % 读取数据
    switch flag
        case 'linear'
            norder = 1;
            % len1 = 25;
            max_lag = 5;    % 最大时延
            load([root, flag, postfix]);   % linear signals
            signals = eval([flag, '_signals100']);
            signals = squeeze(signals(trial, :, :));
        case 'nonlinear'
            max_lag = 5;    % 最大时延
            norder = 2;
            % len1 = 350;
            load([root, flag, postfix]);   % nonlinear signals
            signals = eval([flag, '_signals100']);
            signals = squeeze(signals(trial, :, :));
        case 'longlag_linear'
            max_lag = 10;    % 最大时延
            norder = 1;
            % len1 = 50;
            load([root, flag, postfix]);   % longlag linear signals
            signals = eval([flag, '_signals100']);
            signals = squeeze(signals(trial, :, :));
        case 'longlag_nonlinear'
            max_lag = 10;    % 最大时延
            norder = 2;
            % len1 = 1325;
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

    %% !基于RFOLS 算法的候选项选择器
    [H, Hv] = buildH(normalized_signals, norder, max_lag, id_wgci);
    [Kalman_H, ~, ~, S_No, ERRs] = term_selector(normalized_signals, norder, max_lag, H, threshold);
    terms_chosen = S_No(:, 1:threshold);   % threshold 为候选项的个数
end
