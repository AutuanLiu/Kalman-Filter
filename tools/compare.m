% 2019/2/21
% 比较结果

clear;clc;
load('result.mat');

% 估计系数比较
figure(1);
plot(data(:, 1), '-o');
grid on;
hold on;
plot(data(:, 2), '--*');
hold on;
plot(data(:, 5), '-->');
hold on;
plot(data(:, 8), '--x');
hold on;
plot(data(:, 11), '--v');
xlabel('terms');
ylabel('coef mean');
title('coef estimation comparison');
legend('real coef', 'FROLS', 'bi-Kalman', 'FROLS-bi-Kalman', 'FROLS-SGD');
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 12 7])
saveas(gcf, '../images/coef_com_mean.png');

% 估计系数的误差比较
figure(2);
plot(data(:, 3), '-o');
grid on;
hold on;
plot(data(:, 6), '-*');
hold on;
plot(data(:, 9), '->');
hold on;
plot(data(:, 12), '-x');
xlabel('terms');
ylabel('coef error');
title('coef error comparison');
legend('FROLS', 'bi-Kalman', 'FROLS-bi-Kalman', 'FROLS-SGD');
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 12 7])
saveas(gcf, '../images/coef_com_err.png');

% 估计系数的方差比较
figure(3);
plot(data(:, 4), '-o');
grid on;
hold on;
plot(data(:, 7), '-*');
hold on;
plot(data(:, 10), '->');
hold on;
plot(data(:, 13), '-x');
xlabel('terms');
ylabel('coef variance');
title('coef varience comparison');
legend('FROLS', 'bi-Kalman', 'FROLS-bi-Kalman', 'FROLS-SGD');
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 12 7])
saveas(gcf, '../images/coef_com_var.png');
close all;
