% 使用 FROLS 算法估计系数
% autuanliu@163.com
% 2019/1/13
%

tic;
clear;

data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};

for m=data_type
    flag = m{1, 1};
    switch flag
        case 'linear'
            load(['data/', flag, '_signals5D_noise1.mat']);
            max_lag = 5;    % 最大时延
            threshold = 5;  % 候选项个数
            signals = eval([flag, '_signals']);   % 模型的数据
            [NN, n_ch] = size(signals);
            norder = 1;
            N = NN - max_lag;
            M = nchoosek(max_lag*n_ch + norder, norder) - 1;
        case 'nonlinear'
            load(['data/', flag, '_signals5D_noise1.mat']);
            max_lag = 5;    % 最大时延
            threshold = 5;  % 候选项个数
            signals = eval([flag, '_signals']);   % 模型的数据
            [NN, n_ch] = size(signals);
            norder = 2;
            N = NN - max_lag;
            M = nchoosek(max_lag*n_ch + norder, norder) - 1;
        case 'longlag_linear'
            load(['data/', flag, '_signals5D_noise1.mat']);
            max_lag = 10;    % 最大时延
            threshold = 5;  % 候选项个数
            signals = eval([flag, '_signals']);   % 模型的数据
            [NN, n_ch] = size(signals);
            norder = 1;
            N = NN - max_lag;
            M = nchoosek(max_lag*n_ch + norder, norder) - 1;
        case 'longlag_nonlinear'
            load(['data/', flag, '_signals5D_noise1.mat']);
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
    for ch=n_ch
        flag
        coef_est = zeros(n_ch, M);
        [coff, ~] = FROLS(norder, signals, max_lag, N, threshold, signals(:, ch));
        size(coff)
        coef_est(ch, :) = coff.';
        % 保存估计系数
        f_name = ['data/FROLS_', flag, '_coef.txt'];
        save(f_name, 'coef_est');
    end
end
toc;
