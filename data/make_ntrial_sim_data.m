% Email: autuanliu@163.com
% Date: 2019/2/20
% 生成用于多组实验的仿真数据
%
clear; clc;
%! 初始化设置
npoint = 2048;      % 待研究或者采样的信号长度
nlen = 2100;        % 仿真信号的总长度
nchannel = 5;       % 信号的维度
max_lag = 20;       % 最大时延
err_var = 1;        % 噪音的方差
flag = 1;           % 是否设置噪音, ==0 表示不设置噪音
err_mean = 0;       % 噪音的均值
ntrial = 100;       % 实验次数
% 数据
linear_signals100 = zeros(ntrial, npoint, nchannel);
nonlinear_signals100 = zeros(ntrial, npoint, nchannel);
longlag_linear_signals100 = zeros(ntrial, npoint, nchannel);
longlag_nonlinear_signals100 = zeros(ntrial, npoint, nchannel);

for trial=1:ntrial
    noise = make_noise(nlen, nchannel, err_mean, err_var, flag);
    init = init_signal(max_lag, nchannel);
    x1(1:max_lag, 1) = init(:, 1) + noise(1:max_lag , 1);
    x2(1:max_lag, 1) = init(:, 2) + noise(1:max_lag , 2);
    x3(1:max_lag, 1) = init(:, 3) + noise(1:max_lag , 3);
    x4(1:max_lag, 1) = init(:, 4) + noise(1:max_lag , 4);
    x5(1:max_lag, 1) = init(:, 5) + noise(1:max_lag , 5);

    %%! 线性信号
    for t=(max_lag + 1):nlen  % 信号时域
        x1(t) = 0.95*sqrt(2) * x1(t-1) - 0.9025 * x1(t-2) + noise(t, 1);
        x2(t) = 0.5 * x1(t-2) + noise(t, 2);
        x3(t) = -0.4 * x1(t-3) + noise(t, 3);
        x4(t) = -0.5 * x1(t-2) + 0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 4);
        x5(t) = -0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 5);
    end

    % 设置线性信号并保存仿真数据
    linear_signals = [x1, x2, x3, x4, x5];
    linear_signals = linear_signals((max_lag+1):(max_lag+npoint), :);

    %%! 非线性信号
    for t=(max_lag + 1):nlen  % 信号时域
        x1(t) = 0.95*sqrt(2) * x1(t-1) - 0.9025 * x1(t-2) + noise(t, 1);
        x2(t) = 0.5 * x1(t-2) * x1(t-2) + noise(t, 2);
        x3(t) = -0.4 * x1(t-3) + noise(t, 3);
        x4(t) = -0.5 * x1(t-2) * x1(t-2) + 0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 4);
        x5(t) = -0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 5);
    end

    % 设置非线性信号并保存仿真数据
    nonlinear_signals = [x1, x2, x3, x4, x5];
    nonlinear_signals = nonlinear_signals((max_lag+1):(max_lag+npoint), :);

    %%! 长时延线性信号
    for t=(max_lag + 1):nlen  % 信号时域
        x1(t) = 0.95*sqrt(2) * x1(t-1) - 0.9025 * x1(t-2) + noise(t, 1);
        x2(t) = 0.5 * x1(t-10) + noise(t, 2);
        x3(t) = -0.4 * x1(t-3) + noise(t, 3);
        x4(t) = -0.5 * x1(t-2) + 0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 4);
        x5(t) = -0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 5);
    end

    % 设置线性信号并保存仿真数据
    longlag_linear_signals = [x1, x2, x3, x4, x5];
    longlag_linear_signals = longlag_linear_signals((max_lag+1):(max_lag+npoint), :);

    %%! 长时延非线性信号
    for t=(max_lag + 1):nlen  % 信号时域
        x1(t) = 0.95*sqrt(2) * x1(t-1) - 0.9025 * x1(t-2) + noise(t, 1);
        x2(t) = 0.5 * x1(t-10) * x1(t-10) + noise(t, 2);
        x3(t) = -0.4 * x1(t-3) + noise(t, 3);
        x4(t) = -0.5 * x1(t-2) * x1(t-2) + 0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 4);
        x5(t) = -0.25*sqrt(2) * x4(t-1) + 0.25*sqrt(2) * x5(t-1) + noise(t, 5);
    end

    % 设置长时延非线性信号并保存仿真数据
    longlag_nonlinear_signals = [x1, x2, x3, x4, x5];
    longlag_nonlinear_signals = longlag_nonlinear_signals((max_lag+1):(max_lag+npoint), :);

    % 数据保存
    linear_signals100(trial, :, :) = linear_signals;
    nonlinear_signals100(trial, :, :) = nonlinear_signals;
    longlag_linear_signals100(trial, :, :) = longlag_linear_signals;
    longlag_nonlinear_signals100(trial, :, :) = longlag_nonlinear_signals;
end

% 保存信号数据
save('linear_signals5D_noise100.mat', 'linear_signals100');
save('nonlinear_signals5D_noise100.mat', 'nonlinear_signals100');
save('longlag_linear_signals5D_noise100.mat', 'longlag_linear_signals100');
save('longlag_nonlinear_signals5D_noise100.mat', 'longlag_nonlinear_signals100');

function noise = make_noise(npoint, nchannel, mean_v, variance, flag)
    % flag == 0 表示不加噪音
    if flag == 0
        noise = zeros(npoint, nchannel);
    else
        noise = randn(npoint, nchannel);
        noise = (noise - mean(noise))./std(noise);
        if variance ~= 0
            noise = mean_v + noise * sqrt(variance);
        end
    end
    return;
end

function init = init_signal(max_lag, nchannel)
    init = randn(max_lag, nchannel);
    return;
end