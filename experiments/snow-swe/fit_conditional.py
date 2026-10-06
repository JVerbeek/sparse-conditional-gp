""" Fits sCMOGP and GPFITC on the a holdout year 50 times. Holdout year used for the paper is 2019.

Sources: three daily SNOTEL pillows (output indices 0, 1, 2). Target: Shrine Pass snow course (index 3).
Winters 2011-2020, November to May only, pillows thinned to every 2nd day. One winter of course
readings is held out to show the prediction between visits.
    python fit_conditional.py          # hold out water year 2019
    python fit_conditional.py 2016
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import gpflow as gpf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

tf.random.set_seed(42)
np.random.seed(42)

at_dir = Path(__file__).resolve().parents[2]  # repo root, where atmodel.py lives
sys.path.append(str(at_dir))
from atmodel import ConditionalMOGP, SparseCMOGP

from atlikelihood import TransferLikelihood
from inducing_variable import LMCInducingPointsBase
from sklearn.metrics import mean_squared_error
from matplotlib import color_sequences

# ----------------------------------------------------------------------------- data
d = np.load(Path(__file__).parent / "data" / "nearby.npz")


def month(t):
    return (np.floor((t - np.floor(t)) * 12).astype(int) % 12) + 1


def water_year(t):
    return np.where(month(t) >= 10, np.floor(t) + 1, np.floor(t)).astype(int)


def in_season(t):
    return np.isin(month(t), [11, 12, 1, 2, 3, 4, 5]) & np.isin(water_year(t), list(WINTERS))

def get_kernel():
    #return SpectralMixture(Q=3, active_dims=[0])
    return gpf.kernels.Matern32(active_dims=[0])

HOLDOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
WINTERS = range(2011, 2021)


def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, track_loss_history=True, options={"disp": 50})

rmse_dict = {"sCMOGP":[],"GPFITC":[]}

print("=*"*30, HOLDOUT, "+*"*30)
data_X, data_Y, data_Y_perturbed = [], [], []
n_src = len(d["source_labels"])

colors = color_sequences["Set2"]
source_names = {0:"Copper Mountain", 1:"Vail Mountain", 2:"Fremont Pass"}


for i in range(n_src):
    t, y = d[f"source{i}_t"], d[f"source{i}_y"]
    t, y = t[in_season(t)][::2], y[in_season(t)][::2]
    data_X.append(t.reshape(-1, 1))
    k = gpf.kernels.RBF(lengthscales=0.3, variance=5)
    data_Y_perturbed.append(y.reshape(-1, 1) + np.random.multivariate_normal(np.zeros(len(t)), k(t[:,None])).reshape(-1, 1))
    data_Y.append(y.reshape(-1, 1))

t, y = d["target_t"], d["target_y"]
t, y = t[in_season(t)], y[in_season(t)]
test = water_year(t) == HOLDOUT
Xtest, ytest = t[test].reshape(-1, 1), y[test].reshape(-1, 1)
data_X.append(t[~test].reshape(-1, 1))
data_Y.append(y[~test].reshape(-1, 1))

# Reproduces Figure 6 from the paper
plt.rcParams["font.family"] = "serif"
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
for i, (dx, dy) in enumerate(zip(data_X, data_Y)):
    ax1.plot(dx, dy, color=colors[i], label=source_names[i]+" (pillow)", lw=3) if i < 3 else None
for j, (dx, dy) in enumerate(zip(data_X, data_Y_perturbed)):
    ax2.plot(dx, dy, color=colors[j], lw=3, label=source_names[j]+" (pillow)") if j < 3 else None
ax2.set_ylabel("SWE", fontsize=20)
ax1.set_ylabel("SWE", fontsize=20)
plt.xlabel("year", fontsize=20)

labels = [str(i) for i in np.arange(2011, 2021, 1)]
ax2.tick_params("x", labelbottom=True)
ax2.plot(data_X[-1], data_Y[-1], color="red", lw=0, marker="o", label="Shrine Pass (course)")
ax1.plot(data_X[-1], data_Y[-1], color="red", lw=0, marker="o", label="Shrine Pass (course)")
ax1.legend(fontsize=10, loc="upper right")
ax2.set_title("Perturbed SWE", fontsize=20)
ax1.set_title("Unperturbed SWE", fontsize=20)
plt.xticks(ticks=np.arange(2011, 2021), labels=labels)
plt.tick_params("both", labelsize=15)
plt.savefig("experiments/snow-swe/figures/perturbed_normal.png")
plt.show()


# standardize each output and build the stacked (value, index) arrays
means, stds = [Y.mean() for Y in data_Y], [Y.std() for Y in data_Y]
X = np.vstack([np.hstack((Xi, i * np.ones_like(Xi))) for i, Xi in enumerate(data_X)])
y = np.vstack([np.hstack((((Yi - m) / s), i * np.ones_like(Yi))) for i, (Yi, m, s) in enumerate(zip(data_Y, means, stds))])
print(f"{len(X)} training points, target index {n_src}, held-out winter {HOLDOUT} with {len(Xtest)} readings")
print(f"{len(X[X[:,1] == n_src])} target points")

# Predefine some values common to both models
output_dim = n_src + 1  # Number of outputs
rank = 1  # Rank of W
condition_index = n_src  # the snow course
nIVS = 50 * output_dim
# ======================================================================================[sCMOGP]

rounds = 50

for r in range(rounds):
    condition_index = n_src  # the snow course

    # Base kernel
    k = get_kernel()

    # Coregion kernel
    coreg = gpf.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

    kern = k * coreg

    ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
    iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
    iv_ind = np.concatenate(iv_ind) 
    shuffle = np.random.permutation(np.arange(len(ivs)))
    ivs = ivs[shuffle]
    ivs = np.hstack((ivs, iv_ind))

    model2 = SparseCMOGP((X, y), kernel=kern, exact_target=False, jitter=1e-5, conditioning_index=condition_index, inducing_variable=LMCInducingPointsBase(ivs),
                            likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
    optimize(model2)

    #================================================================================= [GPFITC]
    ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
    iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
    iv_ind = np.concatenate(iv_ind)  
    shuffle = np.random.permutation(np.arange(len(ivs)))
    ivs = ivs[shuffle]
    ivs = np.hstack((ivs, iv_ind))

    k = get_kernel()
    # coregion kernel
    coreg = gpf.kernels.Coregion(
        output_dim=output_dim, rank=rank, active_dims=[1] 
    )

    kern = k * coreg 
    model5 =  gpf.models.GPRFITC((X, y), kernel=kern, inducing_variable=LMCInducingPointsBase(ivs))
    # fit the covariance function parameters
    optimize(model5)
    names = ["sCMOGP", "GPFITC"]
    for i, model in enumerate((model2, model5)):
        Xtst_f = np.hstack((Xtest, condition_index * np.ones((len(Xtest), 1))))
        fmean_test, fvar_test = model.predict_f(Xtst_f)
        fmean_test = fmean_test * stds[condition_index] + means[condition_index]
        try:
            mse = mean_squared_error(ytest, fmean_test[:, 0])
        except ValueError:
            continue
        print(model, mse)
        rmse_dict[names[i]].append(mse)


print([(k, np.mean(v), np.std(v)) for (k, v) in rmse_dict.items()])
import json
# Serialize data into file:
json.dump(rmse_dict, open(f"experiments/snow-swe/2019-{int(nIVS/output_dim)}-perturbed.json", 'w' ) )
#----------------------------------------------------------------------------- predictions
colors = color_sequences["Set2"]
lo, hi = HOLDOUT - 1 + 10 / 12, HOLDOUT + 6 / 12  # plot the held-out winter only
Xtst = Xtest.reshape(-1, 1)

plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
for index in  range(output_dim):
    for name, model, color in zip(["sCMOGP", "GPFITC", "weighted"], models, colors[:5]):
        Xplot = np.linspace(lo, hi, 300)[:,None] if name is "SGPR" else np.hstack((np.linspace(lo, hi, 300)[:, None], index * np.ones((300, 1))))
        Ax, Ay = X[X[:, 1] == index], data_Y[index]
        fmean, fvar = model.predict_f(Xplot)
        fmean = fmean * stds[index] + means[index]  # back to inches
        fvar = fvar * stds[index] ** 2
        
        if index == condition_index:
            Xtst_f = Xtst if name == "SGPR" else np.hstack((Xtst, index * np.ones((len(Xtst), 1))))
            fmean_test, fvar_test = model.predict_f(Xtst_f)
            fmean_test = fmean_test * stds[index] + means[index]
            ax.plot(Xtst, ytest, "r.", ms=12, label="target, held out" if name == "SGPR" else "")
            ax.set_title(f"{label}", fontsize=25)
            try:
                ours_mse = mean_squared_error(ytest, fmean_test[:, 0])
                rmse_dict[name].append(np.sqrt(ours_mse))
            except ValueError:
                pass
            print(np.mean(rmse_dict[name]))
            print(f"{name} held-out readings:", ytest[:, 0].round(1))
            print(f"{name} predicted:        ", fmean_test.numpy()[:, 0].round(1))
            print(f"{name} rmse:", np.sqrt(ours_mse).round(2), "in")

        label = str(d["source_labels"][index]) if index < n_src else str(d["target_label"])
        ax.set_title(f"{label}", fontsize=25)
        m = (Ax[:, 0] >= lo) & (Ax[:, 0] <= hi)
        if name == "SGPR" and index in [2, 3]:
            ax.plot(Ax[m, 0], Ay[m, 0], color="black", alpha=0.5,
                    lw=0 if index == condition_index else 1, label="course" if index == condition_index else "pillow")
        else:
            ax.plot(Ax[m, 0], Ay[m, 0], color="black", alpha=0.5,
                    lw=0 if index == condition_index else 1)
        if index == condition_index:
            ax.plot(Xplot[:, 0], fmean[:,0], color=color, label=f"{name} predictions" if index==output_dim-1 else "", lw=2)
            ax.fill_between(
                Xplot[:, 0],
                (fmean[:,0] - 2 * np.sqrt(fvar)[:, 0]),
                (fmean[:,0] + 2 * np.sqrt(fvar)[:, 0]),
                color=color,
                alpha=0.4,
                lw=2
            )
        ax.tick_params(axis="both", labelsize=20)
        ax.set_xlabel("time (decimal)", fontsize=20)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("SWE [in]", fontsize=20)
fig.legend(fontsize=20)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "figures" / f"fit_conditional_{HOLDOUT}_output{index}_{name}_target.png", dpi=120)
plt.show()