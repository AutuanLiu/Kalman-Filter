% autuanliu@163.com
% 2018年12月10日
% 生成 V 矩阵
% 

function [V] = buildV(n_cnt, lags_sum)
    % n_cnt 当前的非线性次数(包括1)且 n_cnt >= 1
    % lags_sum: 所有信号的延迟和
    % 

    V = [];
    if n_cnt == 1
        V = [1:lags_sum].';
    elseif n_cnt > 1
        V1 = buildV(n_cnt - 1, lags_sum);
        [nrow, ~] = size(V1);
        for t=1:lags_sum
            idx = min(find(V1(:, 1)==t));
            V2 = [ones(nrow-idx+1, 1)*t, V1(idx:nrow, :)];
            V = [V; V2];
        end
    end
    return;
end
