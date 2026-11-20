"""Typography-only substitutions for the retained Week 4 evidence appendix.

The original scientific JSON, numerical values, equations and experiment roles
remain unchanged. This mapping replaces legacy ASCII subscripts in body text.
"""
from copy import deepcopy

SUBSTITUTIONS = {
    'u_pred(t) = t + t(1-t)N_theta and v_pred(t) = t(1-t)N_theta':
        'u<sub>pred</sub>(t) = t + t(1-t)N<sub>θ</sub> and v<sub>pred</sub>(t) = t(1-t)N<sub>θ</sub>',
    'L-BFGS builds an approximate inverse Hessian H_t from recent s_t = theta_(t+1)-theta_t and y_t = g_(t+1)-g_t, and takes direction d_t = -H_t g_t with a line search.':
        'L-BFGS builds an approximate inverse Hessian H<sub>t</sub> from recent s<sub>t</sub> = θ<sub>t+1</sub>-θ<sub>t</sub> and y<sub>t</sub> = g<sub>t+1</sub>-g<sub>t</sub>, and takes direction d<sub>t</sub> = -H<sub>t</sub>g<sub>t</sub> with a line search.',
    's_k is either one common training RMS or each mode RMS.':
        's<sub>k</sub> is either one common training RMS or each mode RMS.',
    'u = psi_y and v = -psi_x':
        'u = ψ<sub>y</sub> and v = -ψ<sub>x</sub>',
    'v_x - u_y = -Laplacian(psi)':
        'v<sub>x</sub> - u<sub>y</sub> = -Laplacian(ψ)',
    'w_i a quadrature weight':
        'w<sub>i</sub> a quadrature weight',
    'A_s is stratum area and n_s its point count.':
        'A<sub>s</sub> is stratum area and n<sub>s</sub> its point count.',
}


def format_pages(pages):
    formatted = deepcopy(pages)
    counts = {key: 0 for key in SUBSTITUTIONS}
    for page in formatted:
        for block in page['blocks']:
            if block['type'] not in ('body', 'small'):
                continue
            for old, new in SUBSTITUTIONS.items():
                counts[old] += block['text'].count(old)
                block['text'] = block['text'].replace(old, new)
    if any(count != 1 for count in counts.values()):
        raise ValueError(f'Unexpected retained-text matches: {counts}')
    return formatted
