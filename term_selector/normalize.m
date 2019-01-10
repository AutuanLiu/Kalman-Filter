% autuanliu@163.com
% 2018年12月13日
% 标准化数据
%

function [normalized_data] = normalize(data, scaler_type)
    % 标准化数据
    %
    % Args:
    %     data (矩阵形式): 未经过标准化的原始数据 (n_point x n_dim)
    %     scaler_type (str): 归一化方式,
    % ! scaler_type set = {'mapminmax', 'zscore'}
    %

    switch scaler_type
        case 'mapminmax'
            normalized_data = mapminmax(data.', 0, 1).';
        case 'zscore'
            normalized_data = zscore(data);
        otherwise
            disp('Not Define!');
    end
    return;
end
