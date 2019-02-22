% autuanliu@163.com
% 2019年2月22日
% 求 WGCI

function [wgci_value, wgci_mean_value, wgci_variance] = WGCI(err, err_all, threshold)
    % err 分子误差，参考 WGCI 的计算公式
    % err_all 分母误差，参考 WGCI 的计算公式
    % threshold: 阈值
    %
    [n_trial, n_ch, ~] = size(err_all);
    wgci_value = zeros(n_trial, n_ch, n_ch);
    for trial = 1:n_trial
        var1 = squeeze(var(err(trial, :, (n_ch + 1):end, :), 0, 3));
        var2 = var(err_all(trial, :, (n_ch + 1):end), 0, 3);
        tmp = log(var1 ./ var2.');
        wgci_value(trial, :, :) = tmp .* (tmp > threshold);
    end
    wgci_mean_value = squeeze(mean(wgci_value));
    wgci_variance = squeeze(var(wgci_value));
    return;
end
