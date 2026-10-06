import numpy as np
import gpflow as gpf 
import matplotlib.pyplot as plt
import time 
import sys
import tensorflow as tf 
tf.random.set_seed(42)
np.random.seed(42)
at_dir = Path(__file__).resolve().parents[2]  # repo root, where atmodel.py lives
from atmodel import SparseCMOGP, ConditionalMOGP
from atlikelihood import TransferLikelihood
from robust_svgp import LMCInducingPointsBase
n_steps = 8
n_tries = 10
output_dim = 2  # Number of outputs
rank = 1  # Rank of W

times = np.zeros((n_steps, n_tries))
MNAME = sys.argv[1] if len(sys.argv) > 1 else sys.exit()
nIVS = 20

def get_model(mname):
    if mname == "SVGP":
        k = gpf.kernels.Matern32(active_dims=[0])
        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )

        kern = k * coreg

        lik = TransferLikelihood(
            source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()
        )

        ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
        iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
        iv_ind = np.concatenate(iv_ind) 
        shuffle = np.random.permutation(np.arange(len(ivs)))
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))

        l1 = gpf.likelihoods.Gaussian()
        l2 = gpf.likelihoods.Gaussian()
        lik = gpf.likelihoods.SwitchedLikelihood(
            [l1 if i != 1 else l2 for i in range(output_dim)]
        )
        # now build the GP model as normal
        m =  gpf.models.SVGP(kernel=kern, likelihood=lik, num_data=len(X), inducing_variable=LMCInducingPointsBase(ivs))

    elif mname == "sCMOGP":
        k = gpf.kernels.Matern32(active_dims=[0])

        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )
        kern = k * coreg
        ivs = np.linspace(0, max(X[:,0]), nIVS).reshape(-1, 1)
        iv_ind = np.concatenate((np.ones((int(nIVS/2),1 )), np.zeros((int(nIVS/2), 1))))  
        shuffle = np.random.permutation(np.arange(len(ivs)))
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))
        ivs = LMCInducingPointsBase(ivs)
        m = SparseCMOGP((X, y), exact_target=False, kernel=kern, jitter=1e-5, inducing_variable=ivs, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
    else:
        k = gpf.kernels.Matern32(active_dims=[0])

        # Coregion kernel
        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )
        kern = k * coreg

        m = ConditionalMOGP((X, y), kernel=kern, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
    return m


for j, NDP in enumerate(np.logspace(6, 14, n_steps, base=2)):
    print("="*10, int(NDP), "="*10)
    gen_k = gpf.kernels.RBF(lengthscales=5, variance=1)
    Xall = np.linspace(0, 50, int(NDP))
    Yall  = np.random.multivariate_normal(np.zeros_like(Xall), gen_k(Xall.reshape(-1, 1)))
    ind_1 = np.arange(0, int(NDP), 1)
    ind_2 = np.linspace(0, NDP-1, int(0.1*NDP)).astype(int)  # Target is 0.1% of source

    X1, y1 = Xall[ind_1], Yall[ind_1] 
    X2 = Xall[ind_2]
    y2 = Yall[ind_2]

    X2 = X2 - X2[0]
    scalar = -1
    y2 = scalar * y2

    y1 = y1.reshape(-1, 1) + np.random.normal(0, 0.1, len(X1)).reshape(-1, 1)
    y2 = y2.reshape(-1, 1) + np.random.normal(0, 0.1, len(X2)).reshape(-1, 1) 
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)

    X = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))
    y = np.vstack((np.hstack((y1, np.zeros_like(y1))), np.hstack((y2, np.ones_like(y2)))))
    for i in range(n_tries):
        print("*"*10, i, "*"*10)
        m = get_model(MNAME)
        if MNAME == "SVGP":
            t = time.time()
            # fit the covariance function parameters
            gpf.optimizers.Scipy().minimize(
                m.training_loss_closure((X, y)),
                m.trainable_variables,
                method="L-BFGS-B",
                options={"maxiter": 50}
            )
            dt_opt = time.time() - t
            times[j, i] = dt_opt
        else:
            t_opt = time.time()
            opt = gpf.optimizers.Scipy().minimize(m.training_loss, m.trainable_variables, method="L-BFGS-B", options={"disp": 50, "maxiter": 50})
            dt_opt = time.time() - t_opt
            times[j, i] = dt_opt
        print("Time:", dt_opt)
        np.savez(f"results-{MNAME}-variable-target.npz", times=times)

