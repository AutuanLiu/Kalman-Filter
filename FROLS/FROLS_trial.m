% 使用 FROLS 算法估计系数 多组测试
% autuanliu@163.com
% 2019/2/20
%

clear;
tic;

%% n_trial 实验
data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};
data_root = '../data/';
n_trial = 100;
is_same = 0;  % 是否使用同一组数据，即是使用一组数据进行100次实验还是使用不同的100组数据进行100次实验

for m=data_type
    flag = m{1, 1};
    coef_est100 = zeros(100, 5, 5);
    ERRs100 = zeros(100, 5, 5);
    terms_chosen100 = zeros(100, 5, 5);
    for trial=1:n_trial
        disp(['### ', int2str(trial)]);
        if is_same == 1
            [coef_est, terms_chosen, ERRs, ~] = FROLS_estimator(data_root, flag, '_signals5D_noise1.mat', 0, 0);
        else
            [coef_est, terms_chosen, ERRs, ~] = FROLS_estimator(data_root, flag, '_signals5D_noise100.mat', trial, 0);
        end
        coef_est5 = zeros(5, 5);
        for row=1:5
            coef_est5(row, :) = coef_est(row, terms_chosen(row, :));
        end
        coef_est100(trial, :, :) = coef_est5;
        terms_chosen100(trial, :, :) = terms_chosen;
        ERRs100(trial, :, :) = ERRs;
    end

    % 求均值方差
    mean_coef = squeeze(mean(coef_est100, 1));
    var_coef = squeeze(var(coef_est100, 1));

    % 保存估计系数
    f_name = [data_root, 'FROLS_', flag, '_est100.mat'];
    save(f_name, 'coef_est100', 'terms_chosen100', 'ERRs100', 'mean_coef', 'var_coef');
end
%%
toc;
