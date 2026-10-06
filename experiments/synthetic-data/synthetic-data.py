import gpflow as gpf
import tensorflow as tf
import numpy as np
tf.random.set_seed(42)
np.random.seed(42)
import sys
from pathlib import Path

at_dir = Path(__file__).resolve().parents[2]  # repo root, where atmodel.py lives
sys.path.append(str(at_dir))
from sklearn.metrics import mean_squared_error
from inducing_variable import LMCInducingPointsBase
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt
import time

from atmodel import ConditionalMOGP, SparseCMOGP
from atlikelihood import TransferLikelihood

def expected_nce(y_true, y_pred, y_var, n_bins=10):
    bins = np.linspace(min(y_pred), max(y_pred), n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    
    rmse_per_bin = []
    rmv_per_bin = []
    
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        
        if np.any(bin_mask):
            rmse = np.sqrt(np.mean((y_true[bin_mask] - y_pred[bin_mask]) ** 2))
            rmse_per_bin.append(rmse)
            
            rmv = np.sqrt(np.mean(y_var[bin_mask]))
            rmv_per_bin.append(rmv)
        else:
            rmse_per_bin.append(0)
            rmv_per_bin.append(1)
    
    nce = (np.array(rmse_per_bin) - np.array(rmv_per_bin)) / np.array(rmv_per_bin)
    return nce

def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, track_loss_history=True, method="L-BFGS-B")
    # plt.plot(res["loss_history"])
    # plt.show()

def get_kernel():
    return gpf.kernels.Matern32(active_dims=[0])

proportion = sys.argv[1] if len(sys.argv) > 1 else 0.05

scmogp_time = np.zeros((10, 2))  # full, sparse
svgp_time = np.zeros((10, 2))  # full, sparse
sgpr_time = np.zeros((10, 2))  # full, sparse
gpfitc_time = np.zeros((10, 2))  # full, sparse

scmogp_mse = np.zeros((10, 2))  # full, sparse
svgp_mse = np.zeros((10, 2))  # full, sparse / 0.01, 0.05, 0.1for i in range(10):
sgpr_mse = np.zeros((10, 2)) 
gpfitc_mse = np.zeros((10, 2)) 

for i in range(10): 
    for n, target_proportion in enumerate([proportion]):
        print("*"*20, i, "*"*20)
        Xs = np.linspace(0, 50, 1000).reshape(-1, 1)
        Xt = np.linspace(0, 50, 1000).reshape(-1, 1)
        fs = np.random.multivariate_normal(np.zeros_like(Xs.flatten()), gpf.kernels.RBF(lengthscales=8, variance=1)(Xs))
        f1 = np.random.multivariate_normal(np.zeros_like(Xs.flatten()), gpf.kernels.RBF(lengthscales=5, variance=1)(Xs))
        f2 = np.random.multivariate_normal(np.zeros_like(Xs.flatten()), gpf.kernels.RBF(lengthscales=1, variance=1)(Xs))
        test_size = int(int(len(Xt)) * 0.2)
        start = int(0.45*len(Xt))
        print(start, start+test_size)
        test_indices = np.arange(start, start + test_size, 1)
        train_indices = [x for x in np.arange(len(Xt)) if x not in test_indices]
        
        ys = (fs + f2 + 0.3*f1 + np.random.normal(0, 0.1, len(Xs))).reshape(-1, 1)
        yt = (fs + 0.2*f2 + np.random.normal(0, 0.1, len(Xt))).reshape(-1, 1) 
        yt_train_full = yt[train_indices]
        Xt_train_full = Xt[train_indices]
        yt_test_full = yt[test_indices]
        Xt_test_full = Xt[test_indices]
        print(ys.shape, yt.shape)

        X_full = np.vstack((np.hstack((Xs, np.zeros_like(Xs))), np.hstack((Xt_train_full, np.ones_like(Xt_train_full)))))
        y_full = np.vstack((np.hstack((ys, np.zeros_like(ys))), np.hstack((yt_train_full, np.ones_like(yt_train_full)))))

        yt_ds = np.array([y for i, y in enumerate(yt) if i % int(1/target_proportion) == 0])
        Xt_ds = np.array([x for i, x in enumerate(Xt) if i % int(1/target_proportion) == 0])

        test_size = int(int(len(Xt_ds)) * 0.25)
        start = int(0.45*len(Xt_ds))
        test_indices = np.arange(start, start + test_size, 1)
        train_indices = [x for x in np.arange(len(Xt_ds)) if x not in test_indices]
        yt_train_ds = yt_ds[train_indices] 
        Xt_train_ds = Xt_ds[train_indices]
        yt_test_ds= yt_ds[test_indices]
        Xt_test_ds = Xt_ds[test_indices]

        X_ds = np.vstack((np.hstack((Xs, np.zeros_like(Xs))), np.hstack((Xt_train_ds, np.ones_like(Xt_train_ds)))))
        y_ds = np.vstack((np.hstack((ys, np.zeros_like(ys))), np.hstack((yt_train_ds, np.ones_like(yt_train_ds)))))

        for j, (X, y, Xtest, ytest) in enumerate([(X_full, y_full, Xt_test_full, yt_test_full), (X_ds, y_ds, Xt_test_ds, yt_test_ds)]):
            output_dim = 2  # Number of outputs
            rank = 1  # Rank of W
            target_index = 1

            # Base kernel
            k = get_kernel()

            # Coregion kernel
            coreg = gpf.kernels.Coregion(
                output_dim=output_dim, rank=rank, active_dims=[1]
            )

            kern = k * coreg

            lik = TransferLikelihood(
                source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()
            )

            nIVS = 50 * output_dim
            ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
            iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
            iv_ind = np.concatenate(iv_ind) 
            shuffle = np.random.permutation(np.arange(len(ivs)))
            ivs = ivs[shuffle]
            ivs = np.hstack((ivs, iv_ind))

            l1 = gpf.likelihoods.Gaussian()
            l2 = gpf.likelihoods.Gaussian()
            lik = gpf.likelihoods.SwitchedLikelihood(
                [l1 if i != target_index else l2 for i in range(output_dim)]
            )
            # Model 2: SVGP
            model2 =  gpf.models.SVGP(kernel=kern, likelihood=lik, num_data=len(X), inducing_variable=LMCInducingPointsBase(ivs))

            t = time.time()
            # fit the covariance function parameters
            gpf.optimizers.Scipy().minimize(
                model2.training_loss_closure((X, y)),
                model2.trainable_variables,
                method="L-BFGS-B",
            )
            dt_svgp = time.time() - t

            ### Model 1: Sparse CMOGP
            # base kernel
            k = get_kernel() 

            # coregion kernel
            coreg = gpf.kernels.Coregion(
                output_dim=output_dim, rank=rank, active_dims=[1] 
            )

            kern = k * coreg 

            ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
            iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
            iv_ind = np.concatenate(iv_ind)  
            ivs = ivs[shuffle]
            shuffle = np.random.permutation(np.arange(len(ivs)))
            ivs = np.hstack((ivs, iv_ind))
            
            model1 = SparseCMOGP((X, y), conditioning_index=target_index, exact_target=False, kernel=kern, jitter=1e-6, inducing_variable=LMCInducingPointsBase(ivs), likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))

            t = time.time()
            optimize(model1)
            dt_scmogp = time.time() - t

            ### Model 3: SGPR
            ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
            iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
            iv_ind = np.concatenate(iv_ind)  
            ivs = ivs[shuffle]
            ivs = np.hstack((ivs, iv_ind))
            if j == 0:
                model3 = gpf.models.SGPR((Xt_train_full[:,0][:,None], yt_train_full[:,0][:,None]), kernel=gpf.kernels.Matern32(), inducing_variable=ivs[:,0][:,None])
            else:
                model3 = gpf.models.SGPR((Xt_train_ds[:,0][:,None], yt_train_ds[:,0][:,None]), kernel=gpf.kernels.Matern32(), inducing_variable=ivs[:,0][:,None])

            t = time.time()
            optimize(model3)
            dt_sgpr = time.time() - t

            ### Model 4: GP-FITC
            k = get_kernel() 

            # coregion kernel
            coreg = gpf.kernels.Coregion(
                output_dim=output_dim, rank=rank, active_dims=[1] 
            )

            kern = k * coreg 

            ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
            iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
            iv_ind = np.concatenate(iv_ind)  
            shuffle = np.random.permutation(np.arange(len(ivs)))
            ivs = ivs[shuffle]
            ivs = np.hstack((ivs, iv_ind))

            model4 =  gpf.models.GPRFITC((X, y), kernel=kern, inducing_variable=LMCInducingPointsBase(ivs))
            t = time.time()
            # fit the covariance function parameters
            optimize(model4)
            dt_gpfitc = time.time() - t
            
            
            plt.rcParams["font.family"] = "serif"
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            for n, (name, model, ax) in enumerate(zip(["sCMOGP", "SVGP", "SGPR", "GPFITC"], (model1, model2, model3, model4), (ax1, ax2, ax3, ax4))):
                fmean, fvar = model.predict_f(np.hstack((Xs, np.ones_like(Xs)))) if n != 2 else model.predict_f(Xs[:,0][:,None])
                ax.plot(Xs, ys, color="C0", label="source")
                if j == 1:
                    ax.plot(Xt_train_ds, yt_train_ds, color="C1", label="train target")
                else:
                    ax.plot(Xt_train_full, yt_train_full, color="C1", label="train target")
                ax.plot(Xt_test_ds, yt_test_ds, marker="o", lw=0, color="black", label="test target")
                ax.plot(Xs, fmean[:,0], color="black", label="predicted f (target)")
                ax.fill_between(
                    Xs[:, 0],
                    (fmean[:,0] - 2 * np.sqrt(fvar[:,0])),
                    (fmean[:,0] + 2 * np.sqrt(fvar[:,0])),
                    lw=2,
                    color="black",
                    alpha=0.2,
                    label = "$\pm 2\sigma$"
                )
                ax.tick_params(axis="both", labelsize=15)
                ax.set_xlabel("" if n in (0, 1) else "x", fontsize=20)
                ax.set_ylabel(""if n in (1, 3) else "y", fontsize=20)
                ax.set_title(name, fontsize=25)
                ax1.sharex(ax3)
                ax2.sharex(ax4)
            plt.legend(fontsize=15)

            mode = "dense" if j == 0 else "sparse"
            plt.suptitle(f"fit with {mode} target data", fontsize=30)
            plt.savefig(f"experiments/toy-example/figures/target{int(target_proportion/0.01)}/dataset-{i}-{mode}")

            # plt.show()

            fmean_test, fvar_test = model1.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
            mse = mean_squared_error(ytest, fmean_test[:,0])
            scmogp_time[i, j] = dt_scmogp
            scmogp_mse[i, j] = mse
            print(model1, "mse", mse, "fit in", dt_scmogp)

            fmean_test, fvar_test = model2.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
            mse = mean_squared_error(ytest, fmean_test[:,0])
            print(model2, "mse", mse, "fit in", dt_svgp)
            svgp_time[i, j] = dt_svgp
            svgp_mse[i, j] = mse

            fmean_test, fvar_test = model3.predict_f(Xtest)
            mse = mean_squared_error(ytest, fmean_test[:,0])
            print(model3, "mse", mse, "fit in", dt_sgpr)
            sgpr_time[i, j] = dt_sgpr
            sgpr_mse[i, j] = mse

            fmean_test, fvar_test = model4.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
            mse = mean_squared_error(ytest, fmean_test[:,0])
            print(model4, "mse", mse, "fit in", dt_gpfitc)
            gpfitc_time[i, j] = dt_gpfitc
            gpfitc_mse[i, j] = mse

np.savez(f"experiments/synthetic-data/results/toy-example-interpolation-{target_proportion}", svgp=svgp_mse, scmogp=scmogp_mse, sgpr=sgpr_mse, gpfitc=gpfitc_mse, scmogp_time=dt_scmogp, svgp_time=dt_svgp, sgpr_time=dt_sgpr, gpfitc_time=dt_gpfitc)