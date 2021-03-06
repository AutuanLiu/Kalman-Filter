% 使用 FROLS 算法估计系数并求 WGCI 多组测试
% autuanliu@163.com
% 2019/2/22
%

clear;
tic;

%% n_trial 实验
data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};
data_root = '../data/';
threshold = 0.01;
n_trial = 100;
ndim = 5;  % 信号的维度
N = 2048;  % 信号的长度
for m=data_type
    flag = m{1, 1};
    y_error1 = zeros(n_trial, ndim, N);
    y_error2 = zeros(n_trial, ndim, N, ndim);
    for trial=1:n_trial
        disp(['### ', int2str(trial)]);
        [~, ~, ~, y_error] = FROLS_estimator(data_root, flag, '_signals5D_noise100.mat', trial, 0);
        y_error1(trial, :, :) = y_error;
        [~, ~, ~, y_error0] = FROLS_estimator(data_root, flag, '_signals5D_noise100.mat', trial, 1);
        y_error2(trial, :, :, :) = y_error0;
    end
    % 求 WGCI
    [wgci_value, wgci_mean_value, wgci_variance] = WGCI(y_error2, y_error1, threshold);
    % 保存结果
    f_name = [data_root, 'FROLS_', flag, '_WGCI100.mat'];
    save(f_name, 'wgci_value', 'wgci_mean_value', 'wgci_variance');
end
%%
toc;
