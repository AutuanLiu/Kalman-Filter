% autuanliu@163.com
% 2018年12月10日
% 基于 FROLS 算法的候选项选择器算法
% 
% 候选项的排列顺序所遵从的主要原则：
% 1. 线性项在前，非线性项在后
% 2. 先从候选信号中选取，再从候选延迟中选取
% 3. 先考虑简单形式再考虑复杂形势
% 4. 统一取最大延迟
% 5. 使用列向量来保存最后的估计系数
% 6. 具体实现思路参看 <<P 矩阵的生成算法.md>>

% 3个信号或通道、非线性次数为 2，则具体的排列顺序为 {共27项}
%! 线性项: 
% y1(t-1), y1(t-2), y2(t-1), y2(t-2), y3(t-1), y3(t-2)             {6}
% 
%! 非线性项: 
% y1^2(t-1), y1(t-1)y1(t-2), y1^2(t-2);                            {3}
% y1(t-1)y2(t-1), y1(t-1)y2(t-2), y1(t-2)y2(t-1), y1(t-2)y2(t-2);  {4}
% y1(t-1)y3(t-1), y1(t-1)y3(t-2), y1(t-2)y3(t-1), y1(t-2)y3(t-2);  {4} 
% y2^2(t-1), y2(t-1)y2(t-2), y2^2(t-2);                            {3}
% y2(t-1)y3(t-1), y2(t-1)y3(t-2), y2(t-2)y3(t-1), y2(t-2)y3(t-2);  {4} 
% y3^2(t-1), y3(t-1)y3(t-2), y3^2(t-2);                            {3}
% 
% 候选项或者模型长度的设置问题处理：
% 当前输出项假如为 y1(t)，那么为了保证延迟项是有意义的，这里仿真数据点的个数应当为大于等于 N+max_lag，其中 N 为实际使用的模型数据点长度, max_lag 为所有候选变量的最大延迟

%! 参考文献
% 1. Billings S A, Chen S, Korenberg M J. Identification of MIMO non-linear systems using a forward-regression orthogonal estimator[J]. International Journal of Control, 1989, 49(6):2157-2189.
% 2. Billings S A. Nonlinear system identification : NARMAX methods in the time, frequency, and spatio-temporal domains[M]. Wiley, 2013.

function [Kalman_H, sparse_H, S, S_No] = term_selector(signals, norder, max_lag, H, threshold)
    % signals: 信号数据 Npoint * ndim(信号个数)
    % norder: 非线性次数
    % max_lag: max lag
    % H: 模型候选项的数据，N*M, 各个输出信号的候选项相同
    % threshold: 算法停止的阈值(这里使用最大项数来约束)
    % 
    % returns:
    % Kalman_H: 用于kalman 滤波器的候选项矩阵
    % sparse_H: 稀疏形式的候选项矩阵
    % S: (selection) term是否选择的 mask 矩阵，ndim*M
    % S_No: (selection) 选择的 term 索引
    % 

    %%! 初始化过程
    [NN, ndim] = size(signals); % signals 信息
    [N, M] = size(H);
    sparse_H = zeros(ndim, N, M);
    Kalman_H = zeros(ndim, N, threshold);
    S = zeros(ndim, M);

    for y_No=1:ndim
        y_tmp = signals(:, y_No);    % 生成当前子系统的数据
        % 为了确保 y(t-max_lag) 有意义，从 y(max_lag+1) 开始计算
        y = y_tmp((max_lag+1):(max_lag+N));
        L = frols_fixed(y, H, threshold);
        S_No(y_No, :) = L;
        % for k=1:M
        %     if L(k) ~= 0
        %         S(y_No, L(k)) = 1;
        %         sparse_H(y_No, :, L(k)) = H(:, L(k));
        %     end
        % end
        idx = sort(L(1, 1:threshold));
        S(y_No, idx) = 1;
        sparse_H(y_No, :, idx) = H(:, idx);
        Kalman_H(y_No, :, :) = H(:, idx);
    end
    return;
end
