import cvxpy as cvx
import numpy as np
import matplotlib.pylab as plt
import scs

# Ensure repeatably random problem data.
np.random.seed(0)

# Generate random data matrix A.
m = 10
n = 10
k = 5
A = np.random.rand(m, k).dot(np.random.rand(k, n))

# Initialize Y randomly.
Y_init = np.random.rand(m, k)

# Ensure same initial random Y, rather than generate new one
# when executing this cell.
Y = Y_init

# Perform alternating minimization.
MAX_ITERS = 30
residual = np.zeros(MAX_ITERS)
for iter_num in range(1, 1 + MAX_ITERS):
    # At the beginning of an iteration, X and Y are NumPy
    # array types, NOT CVXPY variables.

    # For odd iterations, treat Y constant, optimize over X.
    if iter_num % 2 == 1:
        X = cvx.Variable(k, n)
        constraint = [X >= 0]
    # For even iterations, treat X constant, optimize over Y.
    else:
        Y = cvx.Variable(m, k)
        constraint = [Y >= 0]

    # Solve the problem.
    obj = cvx.Minimize(cvx.norm(A - Y * X, 'fro'))
    prob = cvx.Problem(obj, constraint)
    prob.solve(solver=cvx.SCS)

    # if prob.status != cvx.OPTIMAL:
    #     raise Exception("Solver did not converge!")

    print('Iteration %3d, residual norm %7.5f' % (iter_num, prob.value))

    residual[iter_num-1] = prob.value

    # Convert variable to NumPy array constant for next iteration.
    if iter_num % 2 == 1:
        X = X.value
    else:
        Y = Y.value


plt.plot(residual)

# plt.imshow(A)
# plt.figure()
# plt.imshow(Y @ X)
plt.pause(0.01)
k=1




# import cvxpy as cvx
# import numpy
#
# # Problem data.
# m = 30
# n = 20
# numpy.random.seed(1)
# A = numpy.random.randn(m, n)
# b = numpy.random.randn(m)
#
# # Construct the problem.
# x = cvx.Variable(n)
# objective = cvx.Minimize(cvx.sum_squares(A*x - b))
# constraints = [0 <= x, x <= 1]
# prob = cvx.Problem(objective, constraints)
#
# # The optimal objective is returned by prob.solve().
# result = prob.solve()
# # The optimal value for x is stored in x.value.
# print(x.value)
# # The optimal Lagrange multiplier for a constraint
# # is stored in constraint.dual_value.
# print(constraints[0].dual_value)