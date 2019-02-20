% 使用 FROLS 算法估计系数 多组测试
% autuanliu@163.com
% 2019/1/18
%

clear;
tic;

% %% n_trial 实验
% data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};
% data_root = '../data/';
% n_trial = 100;

% for m=data_type
%     flag = m{1, 1};
%     coef_est100 = [];
%     ERRs100 = [];
%     terms_chosen100 = [];
%     for trial=1:n_trial
%         disp(['### ', int2str(trial)]);
%         [coef_est, terms_chosen, ERRs] = FROLS_estimator(data_root, flag);
%         coef_est100 = [coef_est100; coef_est];
%         terms_chosen100 = [terms_chosen100; terms_chosen];
%         ERRs100 = [ERRs100; ERRs];
%     end

%     % 保存估计系数
%     f_name = [data_root, 'FROLS_', flag, '_est100.mat'];
%     save(f_name, 'coef_est100', 'terms_chosen100', 'ERRs100');
% end
% %%
% toc;

%% n_trial 实验
% data_type = {'nonlinear', 'longlag_nonlinear'};
data_type = {'nonlinear'};
data_root = '../data/';

for m=data_type
    flag = m{1, 1};
    [coef_est, terms_chosen, ERRs] = FROLS_estimator(data_root, flag);
    % 保存估计系数
    f_name = [data_root, 'FROLS_', flag, '_est100.mat'];
    save(f_name, 'coef_est', 'terms_chosen', 'ERRs');
end
%%
toc;
