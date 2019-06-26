% 进行 FROLS test
% autuanliu@163.com
% 2019/2/20
%

% clear;

%% n_trial 实验
% data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};
data_type = {'linear', 'nonlinear'};
data_root = '';  % 当前文件夹
n_trial = 50;
is_same = 0; % 是否使用同一组数据，即是使用一组数据进行100次实验还是使用不同的100组数据进行100次实验
n_term = 5;   % 保留的项数
% thr = 0.1;  % 通过检验的阈值 这里分段设置

% 这里为了避免麻烦就进行单步测试
for err_var = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    % 噪音方差2以下的使用0.1作为阈值，高于2的使用0.2作为阈值(当实验组数较大时，这里都设置为0.1，当实验次数为50)
    if err_var <= 2
        thr = 0.1;
    else
        thr = 0.1;
    end
    for ndim = [5, 10]
        for m = data_type
            flag = m{1, 1};
            disp(['data_type: ', flag, '    ndim: ', int2str(ndim), '    err_var: ', sprintf('%2.2f', err_var)])
            coef_est100 = zeros(n_trial, ndim, n_term);
            ERRs100 = zeros(n_trial, ndim, n_term);
            terms_chosen100 = zeros(n_trial, ndim, n_term);

            % 考虑数据的整体有效性
            valid = zeros(n_trial, 1);
            for trial = 1:n_trial
                % disp(['### ', int2str(trial)]);

                if is_same == 1
                    fn = ['_signals', int2str(ndim), 'D_noise1.mat'];
                    [coef_est, terms_chosen, ERRs, ~] = FROLS_estimator4test(data_root, flag, fn, 0, 0);
                else
                    fn = ['_signals', int2str(ndim), 'D_noise100_', sprintf('%2.2f', err_var), '.mat'];
                    [coef_est, terms_chosen, ERRs, ~] = FROLS_estimator4test(data_root, flag, fn, trial, 0);
                end

                coef_est5 = zeros(ndim, n_term);

                for row = 1:ndim
                    coef_est5(row, :) = coef_est(row, terms_chosen(row, :));
                end

                % 验证是否是有效数据，每组数据都要验证
                ret = is_data_valid(terms_chosen, coef_est5, flag, ndim, thr);
                valid(trial) = ret;
                
                coef_est100(trial, :, :) = coef_est5;
                terms_chosen100(trial, :, :) = terms_chosen;
                ERRs100(trial, :, :) = ERRs;
            end
            
            % 生成的所有数据是有效的，否则重新生成数据
            if all(valid)
                disp('是有效数据！');
            else
                disp('不是有效数据！');
            end

            % 求均值
            mean_coef = squeeze(mean(coef_est100, 1));

        end
    end
end


% 辅助函数
function [ret] = is_data_valid(terms_chosen, est_coef, data_type, dim, thr)
    load('../candidate_terms/ground_truth.mat');
    true_corr = eval([data_type, 'corr_true', int2str(dim), 'D']);
    true_coef = eval(['true_coefs', int2str(dim)]);
    true_terms = eval(['con_terms', int2str(dim)]);
    m = length(true_coef);
    [row, col] = size(est_coef);
    est = zeros(1, m);
    step = 1;

    for r = 1:row
        for t = 1:true_terms(r)
            for c = 1:col
                if terms_chosen(r, c) == true_corr(step)
                    est(step) = est_coef(r, c);
                else
                    continue;
                end
            end
            step = step + 1;
        end
    end
    
    ret = all(abs(est - true_coef) < thr);
    return;
end
