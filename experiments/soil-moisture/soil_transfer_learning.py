import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import os
import sys
import tensorflow as tf
import time
tf.random.set_seed(42)
np.random.seed(42)
from pathlib import Path
home = Path.home()
at_dir = Path(__file__).resolve().parents[2]  # repo root, where atmodel.py lives
subdir = at_dir / "experiments" / "soil-moisture" / "Wageningen" ## or whatever folder at-gpflow lives in
sys.path.append(str(at_dir))
sys.path.append(str(subdir))
from itertools import combinations

from inducing_variable import LMCInducingPointsBase

from sklearn.metrics import mean_squared_error
from matplotlib import color_sequences

from atmodel import SparseCMOGP, ConditionalMOGP
from atlikelihood import TransferLikelihood
import gpflow as gpf

BASE = str(subdir)

def get_kernel():
    
    # Base kernel
    k = gpf.kernels.Matern32(active_dims=[0]) + gpf.kernels.Matern12(active_dims=[0])

    # Coregion kernel
    coreg = gpf.kernels.Coregion(
    output_dim=output_dim, rank=rank, active_dims=[1]
    )
    return k * coreg

def get_inducing_vars(number):
    nIVS = number * output_dim
    ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
    iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
    iv_ind = np.concatenate(iv_ind) 
    shuffle = np.random.permutation(np.arange(len(ivs)))
    ivs = ivs[shuffle]
    ivs = np.hstack((ivs, iv_ind))
    return LMCInducingPointsBase(ivs)

def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, track_loss_history=True, options={"disp": 50})
    # plt.plot(res["loss_history"])
    # plt.show()


# Experiment 1: different substations using multiple depths

tuples = [("04", "01"), 
         ("14", "15"),
         ("05", "07"),
         ("09", "10")]


pair_index = sys.argv[1] if len(sys.argv) > 1 else 0
station_indices = [np.repeat(tup, 2) for tup in tuples]
print(station_indices)
depths = np.concatenate([[5, 10] for t in tuples[pair_index]])
substations = []

for station, depth in zip(station_indices[pair_index], depths):

    source = pd.read_csv(BASE+f"/data/station_data/RM_SM_{station}_calibrated.csv")
    source["Measurement Time"] = pd.to_datetime(source["Measurement Time"])
    source.index = source["Measurement Time"]
    source = source.resample("12H").last()

    source['decimal_year'] = source['Measurement Time'].dt.year + (source['Measurement Time'].dt.dayofyear - 1) / 365.25
    source.index = source["decimal_year"]
    substations.append(source)

    plt.plot(source.index, source[f"{depth} cm VWC [m3/m3]"])
plt.show()

### Construct the data
condition_index = 2
data_X = []
data_Y = []

for i, (depth, data) in enumerate(zip(depths, substations)):
    Ay = data[f"{depth} cm VWC [m3/m3]"].to_numpy().reshape(-1, 1) 
    Ax = data.index.to_numpy().reshape(-1, 1)
    Ax = Ax[~np.isnan(Ay)].reshape(-1, 1)
    Ay = Ay[~np.isnan(Ay)].reshape(-1, 1)
    Ay = (Ay - Ay.mean()) / np.std(Ay)
    if i == condition_index:
        n_test = int(0.33 * len(Ax))
        start = int((1/2) * len(Ax))
        end = start + n_test
        Xtest, ytest = Ax[start:end], Ay[start:end]
        print("Dataset contains", len(Ay), f"points per substation in training,", int(0.2*len(Ay)), "in testing")
        Ax, Ay = np.concatenate((Ax[:start], Ax[end:])), np.concatenate((Ay[:start], Ay[end:]))
    data_X.append(np.hstack((Ax, i*np.ones_like(Ax))))
    data_Y.append(np.hstack((Ay, i*np.ones_like(Ay))))

X = np.vstack(tuple(data_X))
y = np.vstack(tuple(data_Y))

niv_list = [10, 50, 100]

result_dict = {"sCMOGP":[], "SVGP":[], "SGPR":[], "GPFITC":[]}
names = list(result_dict.keys())

#------------------------------------------------------------shared vars
output_dim = len(substations) # Number of outputs
rank = 1 # Rank of W

#------------------------------------------------------------CMOGP (outside loop)
# print("started CMOGP at", time.time())
# # Base kernel
# kern = get_kernel()
# model1 = ConditionalMOGP((X, y), kernel=kern, conditioning_index=condition_index, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))

# optimize(model1)

for nIVS in niv_list:
    #------------------------------------------------------------------------sCMOGP
    print("started sCMOGP at", time.time())
    kern = get_kernel()

    scmogp_ivs = get_inducing_vars(nIVS)

    model2 = SparseCMOGP((X, y), conditioning_index=condition_index, exact_target=False, kernel=kern, jitter=1e-5, inducing_variable=scmogp_ivs, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))

    optimize(model2)

    #-----------------------------------------------------------SVGP

    print("started SVGP at", time.time())
    # Base kernel
    kern = get_kernel()

    svgp_ivs = get_inducing_vars(nIVS)

    l1 = gpf.likelihoods.Gaussian()
    l2 = gpf.likelihoods.Gaussian()
    lik = gpf.likelihoods.SwitchedLikelihood(
        [l1 if i != condition_index else l2 for i in range(output_dim)]
    )
    # now build the GP model as normal
    model3 =  gpf.models.SVGP(kernel=kern, likelihood=lik, num_data=len(X), inducing_variable=svgp_ivs)


    # fit the covariance function parameters
    gpf.optimizers.Scipy().minimize(
        model3.training_loss_closure((X, y)),
        model3.trainable_variables,
        method="L-BFGS-B",
    )
    
    #------------------------------------------------------------------------SGPR
    print("started SGPR at", time.time())
    # Base kernel
    k = gpf.kernels.Matern32(active_dims=[0]) + gpf.kernels.Matern12(active_dims=[0])

    shuffle = np.random.permutation(nIVS)
    ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
    ivs = ivs[shuffle]

    l1 = gpf.likelihoods.Gaussian()
    l2 = gpf.likelihoods.Gaussian()
    lik = gpf.likelihoods.SwitchedLikelihood(
        [l1 if i != condition_index else l2 for i in range(output_dim)]
    )

    Xt, yt = X[X[:,1] == condition_index][:,0].reshape(-1, 1), y[y[:,1] == condition_index][:,0].reshape(-1, 1)
    # now build the GP model as normal
    model4 =  gpf.models.SGPR((Xt, yt), kernel=k, likelihood=l1, inducing_variable=ivs[:,0].reshape(-1, 1))

    optimize(model4)

    #------------------------------------------------------------------------GPFITC

    print("started GPFITC at", time.time())
    kern = get_kernel()

    gprfitc_ivs = get_inducing_vars(nIVS)

    l1 = gpf.likelihoods.Gaussian()

    # now build the GP model as normal
    model5 =  gpf.models.GPRFITC((X, y), kernel=kern, likelihood=l1, inducing_variable=gprfitc_ivs)

    gpf.utilities.print_summary(model5)
    # fit the covariance function parameters
    optimize(model5)

    #------------------------------------------------------------------------Plotting block
    # ---------------------------------- Plots for all substations
    plt.rcParams["font.family"] = "serif"
    for name, model in zip(names, [model2, model3, model4, model5]):
        fig, ax = plt.subplots(output_dim, 1, figsize=(8, 12), sharex=True)
        colors = color_sequences["Set2"]
        for index in range(output_dim):
            fig.suptitle(name, fontsize=20)
            Ax, Ay = X[X[:,1] == index], y[y[:,1] == index]
            Xplot = np.hstack((np.linspace(min(X[:,0]), max(X[:,0]), 200)[:,None], index*np.ones((200, 1))))
            fmean, fvar = model.predict_f(Xplot) if name != "SGPR" else model.predict_f(np.linspace(min(X[:,0]), max(X[:,0]), 200)[:,None])
            ax[index].set_title(f"station {station_indices[pair_index][index]} depth {depths[index]}", fontsize=15)
            ax[index].set_xlabel(f"time" if index == output_dim - 1 else "", fontsize=15)
            ax[index].set_ylabel(f"VWC", fontsize=15)
            if index == condition_index:    
                ax[index].plot(Xtest, ytest, "r.", alpha=0.5, label="test data")
            ax[index].plot(Xplot[:,0], fmean[:,0], color=colors[index], label="predictions")
            ax[index].plot(Ax[:,0], Ay[:,0], color=colors[index], marker=".", alpha=0.5, label="station data")
            ax[index].fill_between(
                Xplot[:, 0],
                (fmean[:,0] - 2 * np.sqrt(fvar)[:, 0]),
                (fmean[:,0] + 2 * np.sqrt(fvar)[:, 0]),
                color=colors[index],
                alpha=0.4,
                label="$\pm 2\sigma$"
            )
            if index == condition_index:  
                fmean_test, fvar_test = model.predict_f(np.hstack((Xtest, index*np.ones((len(Xtest), 1))))) if name != "SGPR" else model.predict_f(Xtest.reshape(-1, 1))
                try:   
                    model_mse = mean_squared_error(ytest[:,0], fmean_test[:,0])
                except ValueError:
                    model_mse = -1
                result_dict[name].append(model_mse)
                ax[index].legend(loc="upper right")
        exp_name = "_".join(triples[pair_index])
        plt.tight_layout()
        plt.savefig(f"experiments/soil-moisture/plots-paper/{exp_name}/inducing_{nIVS}/{name}_all_stations")
        plt.tight_layout()
    #---------------------------------- Create four plots for in the paper  
    colors = color_sequences["Set2"]
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    for i, (model, ax) in enumerate(zip([ model2, model3, model4, model5], [ax1, ax2, ax3, ax4])):
        for index in [condition_index]:
            Ax, Ay = X[X[:,1] == index], y[y[:,1] == index]
            Xplot = np.hstack((np.linspace(min(X[:,0]), max(X[:,0]), 200)[:,None], index*np.ones((200, 1))))
            fmean, fvar = model.predict_f(Xplot) if names[i] != "SGPR" else model.predict_f(np.linspace(min(X[:,0]), max(X[:,0]), 200)[:,None])
            if names[i] != "SGPR":
                fmean_test, fvar_test = model.predict_f(np.hstack((Xtest, index*np.ones((len(Xtest), 1)))))
            else:
                fmean_test, fvar_test = model.predict_f(Xtest)

            ax.plot(Ax[:,0], Ay[:,0], color="k", marker=".", lw=0, ms=4, alpha=0.5, label=f"target")
            for j, (mean, var) in enumerate([(fmean, fvar)]):
                ax.plot(Xplot[:,0], mean[:,0], color=colors[index], alpha=0.4, label=f"target predictions")
                ax.fill_between(
                    Xplot[:, 0],
                    (mean[:,0] - 2 * np.sqrt(var)[:, 0]),
                    (mean[:,0] + 2 * np.sqrt(var)[:, 0]),
                    color=colors[index],
                    alpha=0.4,
                    label="$\pm 2\sigma$",
                    linestyle=":" if i == 1 else "-"
                )
            ax.set_xlim(np.min(Ax[:,0]), np.max(Ax[:,0]))
            ax.set_title(names[i], fontsize=15)
            fig.supxlabel("year (decimal)", fontsize=15)
            fig.supylabel(f"RM {station_indices[pair_index][condition_index]} {depths[condition_index]}cm VWC (normalized)", fontsize=15)
            ax.tick_params(axis="both", labelsize=15)
            ax.plot(Xtest, ytest, "b.", ms=4, alpha=0.5, label="target test data")
            if i > -1: 
                ivs = model.inducing_variable.Z.numpy()
                ivs_target = ivs[ivs[:,1] == condition_index] if names[i] != "SGPR" else ivs
                ax.scatter(ivs_target, 2*np.ones_like(ivs_target), marker="|", label="inducing variables") 
            if i == 2:
                ax.legend(ncols=2)
            
    plt.tight_layout()
    plt.savefig(f"experiments/soil-moisture/plots-paper/{exp_name}/target_{nIVS}IVs")
    plt.clf()
    print("Result dict", result_dict)

import json
# Write dictionary to JSON file
with open(f"results/{exp_name}.json", "w") as file:
    json.dump(result_dict, file)
