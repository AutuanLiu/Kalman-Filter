% 使用 FROLS 算法估计系数并实现相关的功能，为计算WGCI等做好准备
% autuanliu@163.com
% 2019/2/22
%

function [coef_est, terms_chosen, ERRs, y_error] = FROLS_estimator(root, flag, postfix, trial, wgci)
    % root: 数据存储的根目录
    % flag: 数据的类型、线性、非线性等
    % postfix: 存储文件名的后缀
    % trial: 当前的 trial
    % wgci: 是否是计算 WGCI 模式
    %
    disp([flag, ' signals calculated!'])
    switch flag
        case 'linear'
            load([root, flag, postfix]);
            max_lag = 5;    % 最大时延
            threshold = 5;  % 候选项个数
            if trial == 0
                signals = eval([flag, '_signals']);   % 模型的数据
            else
                signals = eval([flag, '_signals100']);   % 模型的数据
                signals = squeeze(signals(trial, :, :));
            end
            [NN, n_ch] = size(signals);
            norder = 1;
            N = NN - max_lag;
        case 'nonlinear'
            load([root, flag, postfix]);
            max_lag = 5;    % 最大时延
            threshold = 5;  % 候选项个数
            if trial == 0
                signals = eval([flag, '_signals']);   % 模型的数据
            else
                signals = eval([flag, '_signals100']);   % 模型的数据
                signals = squeeze(signals(trial, :, :));
            end
            [NN, n_ch] = size(signals);
            norder = 2;
            N = NN - max_lag;
        case 'longlag_linear'
            load([root, flag, postfix]);
            max_lag = 10;    % 最大时延
            threshold = 5;  % 候选项个数
            if trial == 0
                signals = eval([flag, '_signals']);   % 模型的数据
            else
                signals = eval([flag, '_signals100']);   % 模型的数据
                signals = squeeze(signals(trial, :, :));
            end
            [NN, n_ch] = size(signals);
            norder = 1;
            N = NN - max_lag;
        case 'longlag_nonlinear'
            load([root, flag, postfix]);
            max_lag = 10;    % 最大时延
            threshold = 5;  % 候选项个数
            if trial == 0
                signals = eval([flag, '_signals']);   % 模型的数据
            else
                signals = eval([flag, '_signals100']);   % 模型的数据
                signals = squeeze(signals(trial, :, :));
            end
            [NN, n_ch] = size(signals);
            norder = 2;
            N = NN - max_lag;
        otherwise
            disp('Not Define!')
    end
    % 估计系数并保存
    coef_est = [];
    ERRs = [];
    terms_chosen = [];
    if wgci == 0
        y_error = zeros(n_ch, NN);
        for ch=1:n_ch
            [coff, y_error1, term_idx, ERR] = FROLS(norder, signals, max_lag, N, threshold, signals(:, ch), 0);
            y_error(ch, :) = y_error1;
            coef_est = [coef_est; coff.'];
            terms_chosen = [terms_chosen; term_idx];
            ERRs = [ERRs; ERR];
        end
    else
        y_error = zeros(n_ch, NN, n_ch);
        for ch=1:n_ch
            % set1 = setdiff(1:n_ch, ch);  % 不考虑自身影响
            set1 = 1:n_ch;  % 考虑自身影响
            for id=set1
                [coff, y_error1, term_idx, ERR] = FROLS(norder, signals, max_lag, N, threshold, signals(:, ch), id);
                y_error(ch, :, id) = y_error1;
                coef_est = [coef_est; coff.'];
                terms_chosen = [terms_chosen; term_idx];
                ERRs = [ERRs; ERR];
            end
        end
    end
    % 保存估计系数
    f_name = [root, 'FROLS_', flag, '_est.mat'];
    save(f_name, 'coef_est', 'terms_chosen', 'ERRs');
    return;
end
