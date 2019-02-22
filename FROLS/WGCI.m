% autuanliu@163.com
% 2019年2月22日
% 求 WGCI

function wgci_value = WGCI(err, err_all, threshold)
    % err 分子误差，参考 WGCI 的计算公式
    % err_all 分母误差，参考 WGCI 的计算公式
    % threshold: 阈值
    %
    n_ch = size(err, 1);
    var1 = squeeze(var(err(:, (n_ch + 1):end, :), 0, 2));
    var2 = var(err_all(:, (n_ch + 1):end), 0, 2);
    wgci_value = log(var1 ./ var2);
    wgci_value = wgci_value .* (wgci_value > threshold);
    return;
end
