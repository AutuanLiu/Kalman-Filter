% autuanliu@163.com
% 2018年12月10日
% 

function [L] = frols_fixed(y, P, threshold)
    % P: 维度大小为 N*M 的候选向量矩阵
    % threshold: 递归停止条件 
    % 所预测的序列 y, 即当前子系统 N * 1
    % 
    % Returns:
    % L: 被选择的候选项下标
    % 
    
    %%! 初始化部分
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
    %%

    % 参看 Billings 2013 年专著
    %%! step 1.
    W = P;
    A(1, 1) = 1;
    ERR = (y.' * W).^2 ./ (sigma * dot(W, W));
    
    % 找出本次迭代应当被选择的候选项
    [C, I] = max(ERR(1, :)); % C为ERR最大值，I为该项的序列值
    chosenERR(1) = C;
    L(1) = I;
    flag(I) = 1;

    Q(:, 1) = W(:, I);
    g(1) = dot(y, Q(:, 1)) / dot(Q(:,1), Q(:, 1));
    %%

    %%! step j.
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
    %%
    
    %! 考虑提前终止迭代的情况
    A = A(1:j, 1:j);
    return;
end
