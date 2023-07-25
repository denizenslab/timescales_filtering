"""Utils for applying RBF interpolation.

Adapted from code for Jain et al., 2020 https://proceedings.neurips.cc/paper/2020/hash/9e9a30b74c49d07d8150c8c83b1ccf07-Abstract.html
"""
import numpy as np
import time

from multiprocessing.pool import ThreadPool
from sklearn.linear_model import Ridge, RidgeCV


def phi(t, tau, epsilon):
    return np.exp(-((epsilon * (t[None] - tau[:, None])) ** 2))


def interpolation_function_train(
    word_time, fine_time, vectors, i, best_epsilon, alphas=np.logspace(-5, 0, 10)
):
    print(f"interpolation_function_train unit {i}")
    P = phi(word_time, word_time, best_epsilon)
    r = RidgeCV(alphas=alphas, fit_intercept=False, store_cv_values=True)
    r.fit(P, vectors[:, i])
    interp_P = phi(word_time, fine_time, best_epsilon)
    return r.predict(interp_P), r.alpha_


def interpolation_function_test(word_time, fine_time, vectors, i, best_epsilon, alpha):
    print(f"interpolation_function_test unit {i}")
    P = phi(word_time, word_time, best_epsilon)
    r = Ridge(alpha=alpha, fit_intercept=False)
    r.fit(P, vectors[:, i])
    interp_P = phi(word_time, fine_time, best_epsilon)
    return np.dot(interp_P, r.coef_)


def get_rbf_interpolation_time(data_time, num_points_per_data_time: int = 25):
    num_timesteps = len(data_time) * num_points_per_data_time
    return np.linspace(np.floor(data_time[0]), np.ceil(data_time[-1]), num_timesteps)


def apply_rbf_interpolation(
    vecs: np.ndarray,
    do_train_rbf: bool,
    fc: float,
    data_time: np.ndarray,
    save_string_addition: str,
    best_alphas_filepath: str = "best_alphas/rbf_best_alphas",
    mean_duration: float = 0.3452519556581881,
):
    """Apply RBF interpolation to [vecs].
    Parameters:
    ----------
    vecs: num_words x num_dims.
    do_train_rbf: if True, determine best alphas.
    fc: midpoint of filter band.
    data_time: num_words list of word presentation times.
    best_alphas_filepath: where to store/load best alphas.
    mean_duration: mean number of seconds per word.
    """
    best_alphas_filepath = best_alphas_filepath + save_string_addition
    epsilon = 1 / (1 / fc * mean_duration)
    epsilon = np.clip(epsilon, 0.1, np.inf)

    interp_time = get_rbf_interpolation_time(data_time)
    nhidden = vecs.shape[1]
    start = time.time()
    if do_train_rbf:
        pool = ThreadPool(processes=10)
        x = pool.map(
            lambda i: interpolation_function_train(
                data_time, interp_time, vecs, i, best_epsilon=epsilon
            ),
            range(nhidden),
        )
        interp_vecs = np.vstack([x[i][0] for i in range(len(x))]).astype(
            np.float64
        )  # num_units x num_timesteps
        best_alphas = np.array([x[i][1] for i in range(len(x))]).astype(np.float64)
        end = time.time()
        print((end - start) / 60)
        np.savez(best_alphas_filepath, best_alphas=best_alphas)
    else:
        best_alphas = np.load(best_alphas_filepath + ".npz")["best_alphas"]
        pool = ThreadPool(processes=5)
        x = pool.map(
            lambda i: interpolation_function_test(
                data_time,
                interp_time,
                vecs,
                i,
                best_epsilon=epsilon,
                alpha=best_alphas[i],
            ),
            range(nhidden),
        )
        interp_vecs = np.vstack(x).astype(np.float64)
        end = time.time()
        print((end - start) / 60)
    return interp_vecs.T
