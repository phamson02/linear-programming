'''
maximize 2x1 + 3x2
subject to 2x1 + x2 <= 8
           x1 + 3x2 <= 9
           x1, x2 >= 0
'''

from ortools.linear_solver import pywraplp

def main():
    # Create the linear solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create the variables x1 and x2
    INF = solver.infinity()
    x1 = solver.NumVar(0, INF, 'x1')
    x2 = solver.NumVar(0, INF, 'x2')

    # Define constraints
    solver.Add(2*x1 + x2 <= 8)
    solver.Add(x1 + 3*x2 <= 9)

    # Define objective func
    solver.Maximize(2*x1 + 3*x2)

    solver.Solve()

    print('Solution:')
    print(f'Objective value: {solver.Objective().Value()}')
    print(f'x1 = {x1.solution_value()}')
    print(f'x2 = {x2.solution_value()}')

if __name__ == '__main__':
    main()