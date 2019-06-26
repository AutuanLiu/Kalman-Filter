% autuanliu@163.com
% 2019/6/25

% ground truth

% 真实候选项
linear5 = {'x1(t-1)', 'x1(t-2)', 'x1(t-2)', 'x1(t-3)', 'x1(t-2)', 'x4(t-1)', 'x5(t-1)', 'x4(t-1)', 'x5(t-1)'};
nonlinear5 = {'x1(t-1)', 'x1(t-2)', 'x1(t-2)*x1(t-2)', 'x1(t-3)', 'x1(t-2)*x1(t-2)', 'x4(t-1)', 'x5(t-1)', 'x4(t-1)', 'x5(t-1)'};
linear10 = {'x1(t-1)', 'x1(t-2)', 'x1(t-2)', 'x2(t-3)', 'x1(t-2)', 'x4(t-4)', 'x9(t-2)', 'x4(t-4)', 'x1(t-1)', 'x1(t-2)', 'x7(t-2)', 'x8(t-3)', 'x9(t-3)', 'x8(t-3)', 'x9(t-3)', 'x7(t-4)'};
nonlinear10 = {'x1(t-1)', 'x1(t-2)', 'x1(t-2)*x1(t-2)', 'x2(t-3)', 'x1(t-2)', 'x4(t-4)', 'x9(t-2)', 'x4(t-4)', 'x1(t-1)*x1(t-2)', 'x1(t-2)', 'x7(t-2)', 'x8(t-3)', 'x9(t-3)', 'x8(t-3)', 'x9(t-3)', 'x7(t-4)'};

% 真实系数
true_coefs5 = [0.95*sqrt(2), -0.9025, 0.5, -0.4, -0.5, 0.25*sqrt(2), 0.25*sqrt(2), -0.25*sqrt(2), 0.25*sqrt(2)];
true_coefs10 = [0.95*sqrt(2), -0.9025, 0.5, 0.9, -0.5, 0.8, -0.4, -0.8, 0.4, -0.4, -0.9, 0.4, 0.3, -0.3, 0.4, -0.75];
con_terms5 = [2, 1, 1, 3, 2];
con_terms10 = [2, 1, 1, 1, 2, 1, 2, 3, 2, 1];

% 加载候选项字典
load('candidate_terms_map.mat');

% 真实候选项位置
for data_type = {'linear', 'nonlinear'}
    for dim = [5, 10]
        data = eval([data_type{1, 1}, 'candidateterms', int2str(dim), 'Dmap']);
        terms = eval([data_type{1, 1}, int2str(dim)]);
        m = length(terms);
        corr_true = zeros(1, m);
        for i=1:m
            corr_true(1, i) = data(terms{1, i});
        end

        eval([data_type{1, 1}, 'corr_true', int2str(dim), 'D', '=', 'corr_true', ';']);
    end

end

% clear 
clear data terms m corr_true i data_type dim ans;

% 保存数据
save('ground_truth.mat');
