% autuanliu@163.com
% 2018年12月10日
% 生成 H 矩阵, 候选项矩阵
% 

function [H, Hv] = buildH(signals, norder, max_lag)
    % signals: 信号数据 Npoint * ndim(信号个数)
    % norder: 非线性次数
    % max_lag: max lag
    % 
    % returns:
    % H: 候选项矩阵
    % Hv: 候选项组合
    % 

    %* 候选向量不考虑误差项的系数，我们认为误差是一个常数
    % 候选向量的个数，-1是减去常数项(也可以看成是误差项的一部分)
    [NN, ndim] = size(signals); % signals 信息
    M = nchoosek(ndim*max_lag + norder, norder) - 1;
    N = NN - max_lag;           % 实际使用的数据点的长度
    H = zeros(N, M);            % 模型候选项的数据
    col_H = 1;                  % 当前 H 矩阵的数据列
    Hv = cell(norder, 1);       % 交叉项在 base 上的索引
    lags_sum = ndim * max_lag;  % 所有信号的延迟之和

    %%! H 矩阵线性部分(base)
    for variable=1:ndim
        for lag = 1:max_lag
            H(:, col_H) = signals((max_lag-lag+1):(max_lag-lag+N), variable);
            col_H = col_H + 1;
        end
    end
    base = H;
    Hv{1} = buildV(1, lags_sum);
    %%

    %%! H 矩阵非线性部分
    % 生成 H 矩阵，H 矩阵将作为 global 变量(为了后续子函数的调用考虑)
    for lv = 2:norder
        tmpV = buildV(lv, lags_sum);
        Hv{lv} = tmpV;
        [row, ~] = size(tmpV);
        for t=1:row
            H(:, col_H) = prod(base(:, tmpV(t, :)), 2);
            col_H = col_H + 1;
        end
    end
    %%

    return;
end
