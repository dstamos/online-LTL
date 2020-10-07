# from numpy.linalg import norm
#
#
# def mean_squared_error(data_settings, x, y_true, w, task_indeces):
#     n_tasks = len(task_indeces)
#     mse = 0
#     for _, task_idx in enumerate(task_indeces):
#         n_points = len(y_true[task_idx])
#         pred = X[task_idx] @ W[:, task_idx]
#
#         mse_temp = norm(y_true[task_idx].ravel() - pred) ** 2 / n_points
#         mse = mse + mse_temp
#
#     performance = mse / n_tasks
#     return performance
