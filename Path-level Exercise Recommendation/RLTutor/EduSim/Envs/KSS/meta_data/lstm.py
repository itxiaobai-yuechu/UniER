import numpy as np

def lstm_cell_forward(x_t, h_prev, c_prev, parameters):
    W_f = parameters["Wf"]
    b_f = parameters["bf"]
    W_i = parameters["Wi"]
    b_i = parameters["bi"]
    W_c = parameters["Wc"]
    b_c = parameters["bc"]
    W_o = parameters["Wo"]
    b_o = parameters["bo"]
    W_y = parameters["Wy"]
    b_y = parameters["by"]

    n_x, m = x_t.shape
    n_y, n_h = W_y.shape

    concat = np.zeros((n_x + n_h, m))
    concat[: n_x, :] = x_t
    concat[n_x:, :] = h_prev

    z_f = sigmoid(np.matmul(W_f, concat) + b_f)
    z_i = sigmoid(np.matmul(W_i, concat) + b_i)
    z_o = sigmoid(np.matmul(W_o, concat) + b_o)
    z = np.tanh(np.matmul(W_c, concat) + b_c)

    c_next = (z_f * c_prev) + (z_i * z)
    h_next = z_o * np.tanh(c_next)
    y_t =sigmoid(np.matmul(W_y, h_next) + b_y)

    cache = (h_next, c_next, h_prev, c_prev, z_f, z_i, z, z_o, x_t, parameters)

    return h_next, c_next, y_t, cache

def lstm_cell_backward(dh_next, dc_next, cache):
    (h_next, c_next, h_prev, c_prev, z_f, z_i, z, z_o, x_t, parameters) = cache

    n_x, m = x_t.shape
    n_h, m = h_next.shape

    gradients = {""}
    return gradients

def lstm_forward(x, h_0, parameters):
    caches = []
    n_x, m ,T_x = x.shape
    n_y, n_a = parameters["W_y"].shape

    a = np.zeros(())
