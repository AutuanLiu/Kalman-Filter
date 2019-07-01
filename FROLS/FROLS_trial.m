% 使用 FROLS 算法估计系数 多组测试
% autuanliu@163.com
% 2019/2/20
%

clear;
tic;

%% n_trial 实验
% data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};
data_type = {'linear', 'nonlinear'};
data_root = '../data/';
n_trial = 1;
is_same = 0;   % 是否使用同一组数据，即是使用一组数据进行100次实验还是使用不同的100组数据进行100次实验
n_term = 5;    % 保留的项数
for err_var = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    for ndim=[5, 10]
        for m=data_type
            flag = m{1, 1};
            disp(['data_type: ', flag, '    ndim: ', int2str(ndim), '    err_var: ', sprintf('%2.2f', err_var)])
            coef_est100 = zeros(n_trial, ndim, n_term);
            ERRs100 = zeros(n_trial, ndim, n_term);
            terms_chosen100 = zeros(n_trial, ndim, n_term);
            for trial=1:n_trial
            % for trial=n_trial
                disp(['### ', int2str(trial)]);
                if is_same == 1
                    fn = ['_signals', int2str(ndim), 'D_noise1.mat'];
                    [coef_est, terms_chosen, ERRs, ~] = FROLS_estimator(data_root, flag, fn, 0, 0);
                else
                    fn = ['_signals', int2str(ndim), 'D_noise100_', sprintf('%2.2f', err_var), '.mat'];
                    [coef_est, terms_chosen, ERRs, ~] = FROLS_estimator(data_root, flag, fn, trial, 0);
                end
                coef_est5 = zeros(ndim, n_term);
                for row=1:ndim
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
            f_name = [data_root, 'FROLS_', int2str(ndim), flag, '_est100_', sprintf('%2.2f', err_var), '.mat'];
            save(f_name, 'coef_est100', 'terms_chosen100', 'ERRs100', 'mean_coef', 'var_coef');
        end
    end
end
%%
toc;
