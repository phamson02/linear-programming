'''
Linear programming solver using the Simplex method.
'''

import numpy as np
import numpy.ma as ma

def to_tableau(c, A, b):
    m, n = A.shape
    tableau = np.zeros((m+1, n+1))
    tableau[:m, :n] = A
    tableau[:m, -1] = b
    tableau[-1, :n] = c
    return tableau

def can_be_improved(tableau):
    print(tableau)
    return np.min(tableau[-1, :-1]) < 0

def get_pivot_position(tableau):
    q = np.argmin(tableau[-1, :-1])
    x = ma.masked_less_equal(tableau[:-1, q], 0)
    p = np.argmin(tableau[:-1, -1]/x)
    return p, q

def pivot_step(tableau, pivot_position):
    p, q = pivot_position
    not_p = [i for i in range(tableau.shape[0]) if i != p]
    tableau[p] /= tableau[p, q]
    tableau[not_p] -= tableau[p].reshape(1,-1) * tableau[not_p, q].reshape(-1,1)
    return tableau

def get_result(tableau):
    result = {'objective': tableau[-1, -1]}
    is_basic = np.count_nonzero(tableau[:,:-1], axis=0) == 1
    solution = np.zeros(tableau.shape[1]-1)
    solution[is_basic] = np.sum(tableau[:, np.where(is_basic)[0]] * tableau[:, -1].reshape(-1,1), axis=0)
    result['solution'] = solution
    return result

def LP_simplex(c, A, b, mode):
    assert mode in ['maximize', 'minimize'], 'maximize or minimize?'
    if mode == 'maximize':
        c = -c
    
    tableau = to_tableau(c, A, b)

    iter = 0
    while can_be_improved(tableau):
        iter += 1
        pivot_position = get_pivot_position(tableau)
        print(f'Iter: {iter}, Pivot: {pivot_position}')
        tableau = pivot_step(tableau, pivot_position)

    return get_result(tableau)

if __name__ == "__main__":
    c = np.array([2, 1, 1, 0, 0])
    A = np.array([[1, 1, 1, 1, 1],
                  [1, 1, 2, 2, 2],
                  [1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 1],])
    b = np.array([5, 8, 2, 5])
    result = LP_simplex(c, A, b, 'maximize')
    print(f'{result}')