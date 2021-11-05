'''
Linear programming solver using the Simplex method.
'''

import numpy as np
import numpy.ma as ma

def to_tableau(c, A, b):
    assert A.shape[1] == c.shape[0], 'Length of c and number of columns of A don\'t match'
    assert A.shape[0] == b.shape[0], 'Length of b and number of rows of A don\'t match'

    m, n = A.shape
    tableau = np.zeros((m+1, n+1))
    tableau[:m, :n] = A
    tableau[:m, -1] = b
    tableau[-1, :n] = c
    return tableau

def can_be_improved(tableau):
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

def get_result(tableau, mode):
    if mode == 'maximize':
        optimal = tableau[-1, -1]
    else:
        optimal = -tableau[-1, -1]
    result = {'objective': optimal}

    basic_vars = get_basic(tableau)
    solution = np.zeros(tableau.shape[1]-1)
    solution[basic_vars] = np.sum(tableau[:, basic_vars] * tableau[:, -1].reshape(-1,1), axis=0)
    result['solution'] = solution

    return result

def get_basic(tableau):
    is_basic = np.count_nonzero(tableau[:,:-1], axis=0) == 1
    return np.where(is_basic)[0]

def cost_row_reduction(tableau):
    is_basic = np.count_nonzero(tableau[:-1,:-1], axis=0) == 1
    basic_vars = np.where(is_basic)[0]
    for q in basic_vars:
        p = np.where(tableau[:-1, q] == 1)[0]
        tableau = pivot_step(tableau, (p, q))
    return tableau

def phase_1(c, A, b):
    A_aux = np.concatenate((A, np.eye(A.shape[0])), axis=1)
    c_aux = np.concatenate((np.zeros(A.shape[1]), np.ones(A.shape[0])), axis=0)
    
    tableau = to_tableau(c_aux, A_aux, b)
    tableau = cost_row_reduction(tableau)

    print('Phase 1:')
    tableau = phase_2(tableau)

    return np.delete(tableau, list(range(A.shape[1], A_aux.shape[1])), axis=1)[:-1]
    
def LP_simplex(c, A, b, mode):
    assert mode in ['maximize', 'minimize'], 'maximize or minimize?'
    if mode == 'maximize':
        c = -c

    tableau = to_tableau(c, A, b)
    if get_basic(tableau).shape[0] < A.shape[0]:
        tableau[:-1] = phase_1(c, A, b)

    tableau = cost_row_reduction(tableau)

    print('Phase 2:')
    tableau = phase_2(tableau)

    return get_result(tableau, mode)

def phase_2(tableau):
    iter = 0
    while can_be_improved(tableau):
        iter += 1
        print(tableau)
        pivot_position = get_pivot_position(tableau)
        print(f'Iter: {iter}, Pivot: {pivot_position}')
        tableau = pivot_step(tableau, pivot_position)
    print(tableau)
    return tableau

if __name__ == "__main__":
    # c = np.array([3, -1, 3, 0, 0, 0])
    # A = np.array([[2, 1, 1, 1, 0, 0],
    #               [1, 2, 3, 0, 1, 0],
    #               [2, 2, 1, 0, 0, 1],])
    # b = np.array([2, 5, 6])
    # result = LP_simplex(c, A, b, 'minimize')

    # c = np.array([3, 2, 0, 0, 0])
    # A = np.array([[2, 1, 1, 0, 0],
    #               [1, 2, 0, 1, 0],
    #               [1,-1, 0, 0, 1]])
    # b = np.array([7, 8, 2])
    # result = LP_simplex(c, A, b, 'maximize')

    # c = np.array([4, 1, 1])
    # A = np.array([[2, 1, 2],
    #               [3, 3, 1],])
    # b = np.array([4, 3])
    # result = LP_simplex(c, A, b, 'minimize')

    c = np.array([2, 1, 1, 0, 0])
    A = np.array([[1, 1, 1, 1, 1],
                  [1, 1, 2, 2, 2],
                  [1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 1]])
    b = np.array([5, 8, 2, 5])
    result = LP_simplex(c, A, b, 'maximize')

    print(f'{result}')