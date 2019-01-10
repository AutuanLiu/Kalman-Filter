% autuanliu@163.com
% 2018年12月10日
% 

NN = 1030;
ndim = 3;          % 信号的维度
N = 1000;          % 用于真是计算的数据点的长度
y1 = zeros(NN, 1); % 声明数据的维度
y2 = zeros(NN, 1);
y3 = zeros(NN, 1);
max_lag = 2;  % 分别对应于各个信号的延迟 lag
norder = 2;        % 非线性最高次数(order)
threshold = 4;  % 进行 FROLS 估计系数时的阈值

% 生成仿真信号
% 为保证信号延迟有意义，前 lag 信号点使用随机数据
y1(1:max_lag) = rand(max_lag, 1);
y2(1:max_lag) = rand(max_lag, 1);
y3(1:max_lag) = rand(max_lag, 1);

for n=3:NN  % 信号时域
    y1(n) = 0.5 * y3(n-1);
    y2(n) = 0.5 * y2(n-1) - 0.3 * y2(n-2) + 0.1 * y3(n-2) + 0.4 * y3(n-1) * y3(n-2);
    y3(n) = 0.3 * y3(n-1) - y3(n-2) - 0.1 * y2(n-2);
end

signals = [y1, y2, y3];

%% 测试
[H, Hv] = buildH(signals, norder, max_lag);
[Kalman_H, sparse_H, S, S_No] = term_selector(signals, norder, max_lag, H, threshold);
