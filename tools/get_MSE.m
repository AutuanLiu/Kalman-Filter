% 获取结果的 MSE 用于评估
% autuanliu@163.com
% 2019/1/23
%

clear;

%% 数据设置
data_type = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'};
data_root = '../data/';
%%

load([data_root, 'linear_signals5D_noise0.mat']);
max_lag = 5;
N = 5000;
% 估计的模型
x1 = linear_signals(1:max_lag, 1);
x2 = linear_signals(1:max_lag, 2);
x3 = linear_signals(1:max_lag, 3);
x4 = linear_signals(1:max_lag, 4);
x5 = linear_signals(1:max_lag, 5);

for t=(max_lag+1):N
    % FROLS
    x1(t) = 1.3391 * x1(t-1) + -0.8927 * x1(t-2) + 0.0178 * x1(t-4) + -0.0198 * x4(t-5) + -0.0342 * x5(t-1);
    x2(t) = 0.5012 * x1(t-2) + 0.0192 * x2(t-1) + -0.0160 * x4(t-4) + -0.0257 * x5(t-4) + 0.0299 * x5(t-5);
    x3(t) = -0.4117 * x1(t-3) + 0.0146 * x1(t-5) + 0.0225 * x2(t-1) + 0.0303 * x4(t-3) + -0.0263 * x5(t-5);
    x4(t) = -0.5092 * x1(t-2) + -0.0341 * x2(t-3) + 0.3387 * x4(t-1) + 0.3546 * x5(t-1) + 0.0393 * x5(t-2);
    x5(t) = 0.0182 * x2(t-3) + 0.0275 * x3(t-1) + -0.3581 * x4(t-1) + 0.0175 * x4(t-5) + 0.3399 * x5(t-1);
end
signal1 = [x1, x2, x3, x4, x5];

for t=(max_lag+1):N
    % Kalman+FROLS
    x1(t) = 1.3353 * x1(t-1) + -0.8898 * x1(t-2) + 0.0175 * x1(t-4) + -0.0213 * x4(t-5) + -0.0352 * x5(t-1);
    x2(t) = 0.5017 * x1(t-2) + 0.0186 * x2(t-1) + -0.0174 * x4(t-4) + -0.0269 * x5(t-4) + 0.0310 * x5(t-5);
    x3(t) = -0.4100 * x1(t-3) + 0.0144 * x1(t-5) + 0.0194 * x2(t-1) + 0.0306 * x4(t-3) + -0.0268 * x5(t-5);
    x4(t) = -0.5111 * x1(t-2) + -0.0330 * x2(t-3) + 0.3346 * x4(t-1) + 0.3493 * x5(t-1) + 0.0373 * x5(t-2);
    x5(t) = 0.0171 * x2(t-3) + 0.0226 * x3(t-1) + -0.3561 * x4(t-1) + 0.0174 * x4(t-5) + 0.3393 * x5(t-1);
end
signal2 = [x1, x2, x3, x4, x5];

for t=(max_lag+1):N
    % bi-Kalman
    x1(t) = 1.3437 * x1(t-1) + -0.9076 * x1(t-2);
    x2(t) = 0.4953 * x1(t-2);
    x3(t) = -0.4340 * x1(t-3);
    x4(t) = -0.4911 * x1(t-2) + 0.3407 * x4(t-1) + 0.3532 * x5(t-1);
    x5(t) = -0.3489 * x4(t-1) + 0.3467 * x5(t-1);
end
signal3 = [x1, x2, x3, x4, x5];

for t=(max_lag+1):N
    % torch+FROLS
    x1(t) = 1.3393 * x1(t-1) + -0.8924 * x1(t-2) + 0.0180 * x1(t-4) + -0.0194 * x4(t-5) + -0.0341 * x5(t-1);
    x2(t) = 0.5016 * x1(t-2) + 0.0193 * x2(t-1) + -0.0157 * x4(t-4) + -0.0258 * x5(t-4) + 0.0299 * x5(t-5);
    x3(t) = -0.4111 * x1(t-3) + 0.0146 * x1(t-5) + 0.0228 * x2(t-1) + 0.0303 * x4(t-3) + -0.0262 * x5(t-5);
    x4(t) = -0.5096 * x1(t-2) + -0.0338 * x2(t-3) + 0.3388 * x4(t-1) + 0.3552 * x5(t-1) + 0.0401 * x5(t-2);
    x5(t) = 0.0180 * x2(t-3) + 0.0270 * x3(t-1) + -0.3586 * x4(t-1) + 0.0179 * x4(t-5) + 0.3400 * x5(t-1);
end
signal4 = [x1, x2, x3, x4, x5];

% MSE
disp('linear FROLS');
sum(sum((signal1-linear_signals).^2))
disp('linear bi-kalman FROLS');
sum(sum((signal2-linear_signals).^2))
disp('linear bi-kalman');
sum(sum((signal3-linear_signals).^2))
disp('linear torch FROLS');
sum(sum((signal4-linear_signals).^2))
