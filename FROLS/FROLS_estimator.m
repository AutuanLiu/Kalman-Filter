% 使用 FROLS 算法估计系数
% autuanliu@163.com
% 2019/1/13
%

function [coef_est, terms_chosen, ERRs] = FROLS_estimator(root, flag)
    disp([flag, ' signals calculated!'])
    switch flag
        case 'linear'
            load([root, flag, '_signals5D_noise1.mat']);
            max_lag = 5;    % 最大时延
            threshold = 5;  % 候选项个数
            signals = eval([flag, '_signals']);   % 模型的数据
            [NN, n_ch] = size(signals);
            norder = 1;
            N = NN - max_lag;
            M = nchoosek(max_lag*n_ch + norder, norder) - 1;
        case 'nonlinear'
            load([root, flag, '_signals5D_noise1.mat']);
            max_lag = 5;    % 最大时延
            threshold = 5;  % 候选项个数
            signals = eval([flag, '_signals']);   % 模型的数据
            [NN, n_ch] = size(signals);
            norder = 2;
            N = NN - max_lag;
            M = nchoosek(max_lag*n_ch + norder, norder) - 1;
        case 'longlag_linear'
            load([root, flag, '_signals5D_noise1.mat']);
            max_lag = 10;    % 最大时延
            threshold = 5;  % 候选项个数
            signals = eval([flag, '_signals']);   % 模型的数据
            [NN, n_ch] = size(signals);
            norder = 1;
            N = NN - max_lag;
            M = nchoosek(max_lag*n_ch + norder, norder) - 1;
        case 'longlag_nonlinear'
            load([root, flag, '_signals5D_noise1.mat']);
            max_lag = 10;    % 最大时延
            threshold = 5;  % 候选项个数
            signals = eval([flag, '_signals']);   % 模型的数据
            [NN, n_ch] = size(signals);
            norder = 2;
            N = NN - max_lag;
            M = nchoosek(max_lag*n_ch + norder, norder) - 1;
        otherwise
            disp('Not Define!')
    end
    % 估计系数并保存
    coef_est = [];
    ERRs = [];
    terms_chosen = [];
    for ch=1:n_ch
        [coff, ~, term_idx, ERR] = FROLS(norder, signals, max_lag, N, threshold, signals(:, ch));
        coef_est = [coef_est; coff.'];
        terms_chosen = [terms_chosen; term_idx];
        ERRs = [ERRs; ERR];
    end
    % 保存估计系数
    f_name = [root, 'FROLS_', flag, '_est.mat'];
    save(f_name, 'coef_est', 'terms_chosen', 'ERRs');
end
