% autuanliu@163.com
% 2018年6月
% FROLS 的通用版本(与数据的维度无关)

% 候选项的排列顺序所遵从的主要原则
% 1. 线性项在前，非线性项在后
% 2. 先从候选信号中选取，再从候选延迟中选取
% 3. 先考虑简单形式再考虑复杂形势
% 4. 统一取最大延迟
% 5. 使用列向量来保存最后的估计系数
% 6. 具体实现思路参看 <<P 矩阵的生成算法.md>>
% 如 候选信号 y1, y2, y3 候选延迟 2(y1), 1(y2), 2(y3)[这里延迟就取2，所以候选延迟有1、2两种]
% 非线性次数为 2，则具体的排列顺序为 {共27项}
% 线性项:   y1(t-1), y1(t-2), y2(t-1), y2(t-2), y3(t-1), y3(t-2)  {6}
% 非线性项: y1^2(t-1), y1(t-1)y1(t-2), y1^2(t-2);  {3}
%           y1(t-1)y2(t-1), y1(t-1)y2(t-2), y1(t-2)y2(t-1), y1(t-2)y2(t-2);  {4}
%           y1(t-1)y3(t-1), y1(t-1)y3(t-2), y1(t-2)y3(t-1), y1(t-2)y3(t-2);  {4}
%           y2^2(t-1), y2(t-1)y2(t-2), y2^2(t-2);  {3}
%           y2(t-1)y3(t-1), y2(t-1)y3(t-2), y2(t-2)y3(t-1), y2(t-2)y3(t-2);  {4}
%           y3^2(t-1), y3(t-1)y3(t-2), y3^2(t-2);  {3}
%
% 候选项或者模型长度的设置问题处理
% 当前输出项假如为 y1(t)，那么为了保证延迟项是有意义的，这里仿真数据点的个数应当为大于等于 N+max_lag
% 其中 N 为实际使用的模型数据点长度, max_lag 为所有候选变量的最大延迟

% 参考文献
% 1. Billings S A, Chen S, Korenberg M J. Identification of MIMO non-linear systems using
% a forward-regression orthogonal estimator[J]. International Journal of Control, 1989, 49(6):2157-2189.
% 2. Billings S A. Nonlinear system identification : NARMAX methods in the time, frequency,
% and spatio-temporal domains[M]. Wiley, 2013.

function [coff, yerror] = FROLS3(norder, signals, lags, N, threshold, y)
    % 调用的函数： generateH、frols_fixed2
    % generateH 和 frols_fixed2 都是该主函数的辅助函数
    % norder: 非线性次数
    % signals: 信号数据 Npoint * ndim(信号个数)
    % lags: 各个对应信号的延迟 1 * ndim(信号个数)
    % N: 实际使用的数据点的长度
    % threshold: 算法停止的阈值
    % y: 当前的输出信号或者对应信号(输出子系统) NN * 1
    % coff: 对应于候选项的系数, 前 sum(lags) 行为线性项的系数，其余为非线性项系数
    %
    % 全局变量
    % H: 存储候选项的矩阵(候选项的排列顺序如上述讨论)，维度 N * M(信号长度 * 候选项个数)
    %
    global H lags_sum Hv; % 内部设置全局变量
    lags_sum = sum(lags);
    % 候选向量不考虑误差项的系数，我们认为误差是一个常数
    % 候选向量的个数，-1是减去常数项(也可以看成是误差项的一部分)
    M = nchoosek(lags_sum + norder, norder) - 1;
    [NN, ndim] = size(signals);
    % 估计系数 实际上是按照真实位置把g重新组合了一下

    % 初始化过程
    max_lag = max(lags);
    Hv = cell(norder, 1);
    coff = zeros(M, 1);    % 算法估计的系数
    yerror = zeros(NN, 1); % 算法估计的误差
    H = zeros(N, M);       % 模型候选项的数据
    col_H = 1;             % 当前 H 矩阵的数据列
    % 生成当前子系统的数据
    % 为了确保 y(t-max_lag) 有意义，从 y(max_lag+1) 开始计算
    y = y((max_lag+1):(max_lag+N));

    % H 矩阵线性部分(base)
    for variable=1:ndim
        for lag = 1:lags(variable)
            H(:, col_H) = signals((max_lag-lag+1):(max_lag-lag+N), variable);
            col_H = col_H + 1;
        end
    end
    base = H(:, 1:(col_H-1));
    Hv{1} = buildV(1);

    % H 矩阵非线性部分
    % 生成 H 矩阵，H 矩阵将作为 global 变量(为了后续子函数的调用考虑)
    for lv = 2:norder
        tmpV = buildV(lv);
        Hv{lv} = tmpV;
        [row, ~] = size(tmpV);
        for i=1:row
            H(:,col_H) = prod(base(:, tmpV(i, :)), 2);
            col_H = col_H + 1;
        end
    end

    % 递归估计系数
    [L, raw_g, A] = frols_fixed(y, H, threshold);
    coff(L(L~=0)) = A \ raw_g(1:size(A, 1));

    % 计算误差项 Z=P*\Theat+E, 所以 E=Z-P*\Theta
    yerror((1 + max_lag):(max_lag + N)) = y - H * coff;
    return;
end

% 辅助函数 1, 生成 V 矩阵
function [V] = buildV(n_cnt)
    % n_cnt 当前的非线性次数(包括1)且 n_cnt >= 1
    global lags_sum;
    V = [];
    if n_cnt == 1
        V = [1:lags_sum].';
    elseif n_cnt > 1
        V1 = buildV(n_cnt - 1);
        [nrow, ~] = size(V1);
        for i=1:lags_sum
            idx = min(find(V1(:, 1)==i));
            V2 = [ones(nrow-idx+1, 1)*i, V1(idx:nrow, :)];
            V = [V; V2];
        end
    end
    return;
end

% 辅助函数 2 计算系数
function [L, g, A] = frols_fixed(y, P, threshold)
    % P: 维度大小为 N*M 的候选向量矩阵
    % L: 被选择的候选项下标
    % g, A: 递归计算的 g, A 向量
    % threshold: 递归停止条件
    % 所预测的序列 y, 即当前子系统 N * 1
    % 最传统的采用阈值的 FROLS 算法(可以使用已经选择的候选项的个数作为终止条件)
    % 初始化部分
    [N, M] = size(P);
    W = zeros(N, M);         % 辅助候选向量矩阵满足 W=P*A^{-1}
    Q = zeros(N, M);         % 从候选向量矩阵中经过筛选后的正交矩阵
    ERR = zeros(1, M);       % 对应于M个候选项的误差减少比率
    A = zeros(M, M);         % 辅助分解矩阵满足 P^T*P=A^T*D*A
    chosenERR = zeros(1, M); % 每次被选择项的ERR值所组成的向量
    L = zeros(1, M);         % 每次选了第几项
    g = zeros(M, 1);         % 估计所得的辅助系数向量满足 g=A*\Theta
    sigma = dot(y, y);       % 向量y的内积
    flag = zeros(1, M);      % 当前候选项是否被选中

    % 参看 Billings 2013 年专著
    %% step 1.
    W = P;
    A(1, 1) = 1;
    ERR = (y.' * W).^2 ./ (sigma * dot(W, W));

    % 找出本次迭代应当被选择的候选项
    [C, I] = max(ERR(1, :)); % C为ERR最大值，I为该项的序列值
    chosenERR(1) = C;
    L(1) = I;
    I
    flag(I) = 1;
    Q(:, 1) = W(:, I);
    g(1) = dot(y, Q(:, 1)) / dot(Q(:,1), Q(:, 1));

    %% step j.
    % 这里最大的迭代步骤不可能大于所有候选项的个数，所以这里默认设置最大的迭代步骤是M
    for j=2:M
        for i=1:M % 对于所有候选项
            if flag(1, i) == 0 % 如果还未被筛选
                % 正交化
                temp = 0;
                for k=1:(j-1)
                    temp = temp + Q(:, k) * dot(P(:, i), Q(:, k)) / dot(Q(:, k), Q(:, k));
                end
                W(:, i) = P(:, i) - temp;
                ERR(j, i) = dot(y, W(:, i))^2 / (sigma * dot(W(:, i), W(:, i))); % 计算ERR
            end
        end

        % 继续选择候选项
        [C, I] = max(ERR(j, :));
        chosenERR(j) = C;
        L(j) = I;
        I
        flag(I) = 1;
        Q(:, j) = W(:, I);
        g(j) = dot(y, Q(:, j)) / dot(Q(:, j), Q(:, j));
        A(j, j) = 1;
        for k=1:(j-1)
            A(k, j) = dot(P(:, I), Q(:, k)) / dot(Q(:, k), Q(:, k));
        end

        % 使用 ERR 和作为迭代终止条件
        % if sum(chosenERR(1, :)) > threshold
        %     break;
        % end

        % 使用 已经选取的候选项个数作为迭代终止条件
        if j==threshold
            break;
        end

        % 使用 ESR 作为迭代终止条件
        % ESR = 1 - sum(chosenERR);
        % if ESR < threshold
        %     break;
        % end
    end
    % 考虑提前终止迭代的情况
    A = A(1:j, 1:j);
    return;
end
