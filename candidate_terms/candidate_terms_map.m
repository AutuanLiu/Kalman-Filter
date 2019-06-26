% autuanliu@163.com
% 2019/6/25

% 加载数据
load('candidate_terms.mat')

for data_type={'linear', 'nonlinear'}
    for dim=[5, 10]
        c = containers.Map;
        data = eval([data_type{1, 1}, 'candidateterms', int2str(dim), 'D']);
        [m, ~] = size(data);
        for i=1:m
            c(data(i)) = i;
        end
        eval([data_type{1, 1}, 'candidateterms', int2str(dim), 'Dmap', '=', 'c', ';']);
    end
end

% clear 无关变量
clear data_type dim c i m data;

% 保存字典
save('candidate_terms_map.mat');

