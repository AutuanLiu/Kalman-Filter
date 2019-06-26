% 用于对数据进行平稳性检验
function ret = unit_root_test(data)
    ndim = size(data, 2);
    % 一开始假设序列三平稳的
    ret = true(1, ndim);
    for dim=1:ndim
        % 1 表示序列是平稳的
        a = adftest(data(:, dim));
        % 0 表示序列是平稳的
        b = kpsstest(data(:, dim));
        % 1 表示序列是平稳的，只要有一个检验满足平稳就说明其平稳
        ret(dim) =  a || ~b;
    end
    return;
end