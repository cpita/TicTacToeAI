import numpy as np


def forward_prop(W_list, b_list, x):
    """Makes a forward pass on the model. Use tanh as the activation for all layers
    :param W_list: List of the model's W parameters
    :type W_list: list[np.array]
    :param b_list: List of the model's b parameters
    :type W_list: list[np.array]
    :param x: Numpy array to make the forward pass on. Shape must be (9, None)
    :type x: np.array
    :return: Final result obtained by the forward pass
    :rtype: np.array
    """
    for W, b in zip(W_list, b_list):
        x = np.tanh(np.dot(W, x) + b)
    return x


def backprop(W_list, b_list, x, v_target):
    """Makes a backward pass on the model
    :param W_list: List of the model's W parameters
    :type W_list: list[np.array]
    :param b_list: List of the model's b parameters
    :type W_list: list[np.array]
    :param x: Numpy array of inputs to make the backward pass on. Shape must be (9, None)
    :type x: np.array
    :param v_target: Numpy array of labels to make the backward pass on. Shape must be (None,)
    :type v_target: np.array
    :return grad_W[::-1]: List of gradients wrt W_list
    :rtype grad_W[::-1]: list[np.array]
    :return grad_b[::-1]: List of gradients wrt b_list
    :rtype grad_b[::-1]: list[np.array]
    """
    Z_list = [None]
    A_list = [x]
    A = x
    for W, b in zip(W_list, b_list):
        Z = np.dot(W, A) + b
        A = np.tanh(Z)
        Z_list.append(Z)
        A_list.append(A)

    grad_W = []
    grad_b = []
    for j in range(len(v_target)):
        dJ_dW_list = []
        dJ_db_list = []
        dv_dz = (1 - A_list[-1][:, j, None] ** 2)
        dv_dW_last = dv_dz * A_list[-2][:, j].T
        dv_db_last = dv_dz
        dJ_dv = (A_list[-1][:, j] - v_target[j])
        dJ_dW_list.append(dv_dW_last * dJ_dv)
        dJ_db_list.append(dv_db_last * dJ_dv)

        for i in list(range(len(W_list) - 1))[::-1]:
            tmp1 = (1 - A_list[i + 1][:, j, None] ** 2)
            tmp2 = W_list[i + 1].T * tmp1
            dv_dz = np.dot(tmp2, dv_dz)
            dv_dW = np.dot(dv_dz, A_list[i][:, j, None].T)
            dJ_dW_list.append(dv_dW * dJ_dv)
            dJ_db_list.append(dv_dz * dJ_dv)
        if len(grad_W) == 0:
            grad_W = dJ_dW_list
            grad_b = dJ_db_list
        else:
            for i in range(len(grad_W)):
                grad_W[i] = grad_W[i] + dJ_dW_list[i]
                grad_b[i] = grad_b[i] + dJ_db_list[i]

    return grad_W[::-1], grad_b[::-1]



