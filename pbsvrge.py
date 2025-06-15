# %%
import numpy as np
from scipy.special import expit  # sigmoid
from tqdm import trange

def powerball(v, gamma):
    """Element-wise Powerball mapping: sign(v) * |v|^gamma"""
    return np.sign(v) * np.abs(v) ** gamma

def logistic_loss(w, X, y, lam):
    z = X @ w
    loss = np.mean(np.log(1 + np.exp(-y * z)))
    reg = lam * np.sum(w**2 / (1 + w**2))
    return loss + reg

def grad_logistic(X, y, w):
    z = X @ w
    prob = expit(-y * z)
    if hasattr(X, 'multiply'):  # sparse matrix
        grad = X.multiply(-(y * prob)[:, np.newaxis])
        return np.mean(grad, axis=0).A1  # .A1 to convert to 1D array
    else:
        grad = -(y * prob)[:, np.newaxis] * X
        return np.mean(grad, axis=0)

def grad_regularizer(w):
    return (2 * w) / (1 + w**2)**2

def pbsvrge(X, y, lam, w0, K, S, b, eta_list, gamma):
    n, d = X.shape
    w = w0.copy()
    history = []
    t = 0
    passes = 1

    for s in trange(S):
        w_tilde = w.copy()
        full_grad = grad_logistic(X, y, w_tilde) + lam * grad_regularizer(w_tilde)
        w_k = w_tilde.copy()

        for k in trange(K):
            indices = np.random.choice(n, b, replace=False)
            Xb = X[indices]
            yb = y[indices]

            grad_wk = grad_logistic(Xb, yb, w_k) + lam * grad_regularizer(w_k)
            grad_tilde = grad_logistic(Xb, yb, w_tilde) + lam * grad_regularizer(w_tilde)

            v = grad_wk - grad_tilde + full_grad
            eta = eta_list[k] if isinstance(eta_list, list) else eta_list

            w_k -= eta * powerball(v, gamma)

            t += b

            if t >= passes * n / 10: 
                history.append(logistic_loss(w_k, X, y, lam))
                passes += 1

        w = w_k.copy()

    return w, history


# %%
# Use deterministic gradient descent to compute w*
def adam_logistic(X, y, lam, w0, eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000, tol=1e-8):
    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = []
    for t in range(1, max_iter + 1):
        grad = grad_logistic(X, y, w) + lam * grad_regularizer(w)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w_new = w - eta * m_hat / (np.sqrt(v_hat) + eps)
        loss = logistic_loss(w_new, X, y, lam)
        history.append(loss)
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w


# %%
from sklearn.datasets import load_svmlight_file

dataset_paths = {
    "a8a": './dataset/a8a.txt',
    "ijcnn1": './dataset/ijcnn1',
    "news20": './dataset/news20.binary',
    "covtype": './dataset/covtype.libsvm.binary'
    # "MNIST": './dataset/MNIST',  # Not included
    # "cifar": './dataset/cifar'   # Not included
}


import pickle

def find_optimal():
    for data, path in dataset_paths.items():
        X, y = load_svmlight_file(path)
        y = y * 1.0  # convert to float
        w0 = np.zeros(X.shape[1])
        lam = 0.1
        w_star[data] = adam_logistic(X, y, lam, w0, eta=0.01, max_iter=2000)
    with open('w_star.pkl', 'wb') as f:
        pickle.dump(w_star, f)
    return w_star


# %%
import matplotlib.pyplot as plt
def train(para_type):
    # para_type = 'gamma'  # 'eta', 'b', 'gamma'
    if para_type in ['eta', 'b']:
        selected_items = [(k, v) for k, v in dataset_paths.items() if k in ['a8a', 'ijcnn1']]
    else:
        selected_items = list(dataset_paths.items())
    for data, path in selected_items:
        X, y = load_svmlight_file(path)
        y = y * 1.0  # convert to float
        w0 = np.zeros(X.shape[1])
        lam = 0.1
        S = 10
        b = 10
        K = int(np.ceil(3 * X.shape[0] / b))
        # S = 30
        # Store sweep results
        results = []

        if data == "a8a" or data == "ijcnn1": eta = 0.01 
        else : eta = 0.1
        
        # Time each parameter sweep and print output        
        if para_type == 'gamma':
            print(f"Training with gamma sweep for dataset: {data}")
            for gamma_val in [0.0, 0.2, 0.4, 0.6, 0.9, 1.0]:
                start_time = time.time()
                w_final, hist = pbsvrge(X, y, lam, w0, K, S, b=b, eta_list=eta, gamma=gamma_val)
                elapsed = time.time() - start_time
                results.append({'gamma': gamma_val, 'history': hist, 'final_loss': hist[-1], 'time': elapsed})
                print(f"Dataset={data}, Gamma={gamma_val}, Final loss={hist[-1]}, Time={elapsed:.2f}s")
            
        elif para_type == 'eta':
            print(f"Training with eta sweep for dataset: {data}")
            for eta_val in [0.001, 0.01, 0.1, 1.0]:
                start_time = time.time()
                w_final, hist = pbsvrge(X, y, lam, w0, K, S, b=b, eta_list=eta_val, gamma=0.9)
                elapsed = time.time() - start_time
                results.append({'eta': eta_val, 'history': hist, 'final_loss': hist[-1], 'time': elapsed})
                print(f"Dataset={data}, Eta={eta_val}, Final loss={hist[-1]}, Time={elapsed:.2f}s")

        elif para_type == "b":
            print(f"Training with batch size sweep for dataset: {data}")
            for b_val in [10, 20, 40, 100]:
                K = int(np.ceil(3 * X.shape[0] / b_val))
                start_time = time.time()
                w_final, hist = pbsvrge(X, y, lam, w0, K, S, b=b_val, eta_list=eta, gamma=0.9)
                elapsed = time.time() - start_time
                results.append({'b': b_val, 'history': hist, 'final_loss': hist[-1], 'time': elapsed})
                print(f"Dataset={data}, Batch size={b_val}, Final loss={hist[-1]}, Time={elapsed:.2f}s")

        else:
            raise ValueError
        
        strs = "$" + ("\\" if para_type != "b" else "") + f"{para_type}$"
        for res in results:
            plt.plot(np.linspace(0, len(res['history']) / 10, len(res['history'])), np.array(res['history']) - logistic_loss(w_star[data], X, y, lam), label=rf'{strs}={res[para_type]}')
        plt.yscale('log')
        plt.xlabel('Passes over data')
        plt.ylabel('Objective Gap')
        plt.legend()
        plt.title(rf'PB-SVRGE Convergence for Different {strs}')
        plt.grid(True)
        plt.savefig(f'./img/pbsvrge_{para_type}_{data}.png')
        # plt.show() # blocks process
        plt.close()
        # break

# %%
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PB-SVRGE Parameter Sweeping")
    parser.add_argument('--para_type', type=str, required=True, choices=['gamma', 'eta', 'b'],
                        help="Parameters: gamma, eta or b")
    args = parser.parse_args()
    w_star = {}
    # Load from pickle
    try:
        with open('w_star.pkl', 'rb') as f:
            w_star = pickle.load(f)
    except FileNotFoundError:
        w_star = find_optimal()
        with open('w_star.pkl', 'wb') as f:
            pickle.dump(w_star, f)
    train(args.para_type)

