types = {'linear_signals', 'nonlinear_signals', 'longlag_linear_signals', 'longlag_nonlinear_signals'};
ndim = size(types, 2);
% 一开始假设序列三平稳的
ret = true(1, ndim);
for dim=1:ndim
    ret(dim) = all(unit_root_test(eval(types{dim})));
end

% 输出最终的检验结果
for dim=1:ndim
    if ret(dim)
        fprintf('%25s 是 平稳序列\n', types{dim});
    else
        fprintf('%25s 不是 平稳序列\n', types{dim});
    end    
end
