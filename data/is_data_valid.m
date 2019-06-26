isvalid
% autuanliu@163.com
% 2019/6/25

function [ret] = is_data_valid(terms_chosen, est_coef, data_type, dim, thr)
    load('../candidate_terms/ground_truth.mat');
    true_corr = eval([data_type, 'corr_true', int2str(dim), 'D']);
    true_coef = eval(['true_coefs', int2str(dim)]);
    true_terms = eval(['con_terms', int2str(dim)]);
    m = length(true_coef);
    [row, col] = size(est_coef);
    est = zeros(1, m);
    step = 1;

    for r = 1:row
        for t = 1:true_terms(r)
            for c = 1:col
                if terms_chosen(r, c) == true_corr(step)
                    est(step) = est_coef(r, c);
                else
                    continue;
                end
            end

            step = step + 1;
        end
    end

    ret = all(abs(est - true_coef) < thr);
    return;
end
