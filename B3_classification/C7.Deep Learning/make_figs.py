import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#003578", "#E0761B", "#3C8D5A"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------- activation functions
t = np.linspace(-6, 6, 400)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
ax[0].plot(t, 1/(1+np.exp(-t)), color=BLUE, lw=2, label="logistic")
ax[0].plot(t, np.tanh(t), color=ORANGE, lw=2, label="tanh")
ax[0].set_title("Saturating activations"); ax[0].legend(frameon=False)
ax[1].plot(t, (1/(1+np.exp(-t)))*(1-1/(1+np.exp(-t))), color=BLUE, lw=2, label="logistic$'$")
ax[1].plot(t, 1-np.tanh(t)**2, color=ORANGE, lw=2, label="tanh$'$")
ax[1].set_title("...and their derivatives"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig("figs/saturation.pdf"); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
ax[0].plot(t, np.clip(t, 0, None), color=GREEN, lw=2); ax[0].set_title("ReLU")
ax[1].plot(t, np.clip(t, .1*t, None), color=GREEN, lw=2); ax[1].set_title("Leaky ReLU ($\\alpha=0.1$)")
ax[2].plot(t, np.log(1+np.exp(t)), color=GREEN, lw=2); ax[2].set_title("Softplus")
fig.tight_layout(); fig.savefig("figs/activations.pdf"); plt.close(fig)

# ---------------------------------------------------- GD vs SGD schematic
rng = np.random.default_rng(3)
n = 60
gd = 2.3*np.exp(-np.arange(n)/22) + .25
sgd = 2.3*np.exp(-np.arange(n)/7) + .25 + rng.normal(0, .035, n)*np.exp(-np.arange(n)/30)
fig, ax = plt.subplots(figsize=(5.2, 2.9))
ax.plot(gd, color=BLUE, lw=2, label="Gradient descent (1 update / epoch)")
ax.plot(sgd, color=ORANGE, lw=2, label="SGD, minibatches (K/B updates / epoch)")
ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figs/gd_vs_sgd.pdf"); plt.close(fig)

# ---------------------------------------------------- parameter count
fig, ax = plt.subplots(figsize=(5.0, 2.6))
names = ["Conv $3\\times3$,\n3$\\to$6 channels", "Dense layer,\nsame output size"]
vals = [168, 3*32*32*6*32*32 + 6*32*32]
bars = ax.barh(names, vals, color=[GREEN, BLUE], height=.5)
ax.set_xscale("log"); ax.set_xlabel("number of parameters (log scale)")
ax.set_xlim(50, vals[1]*60)
for b, v in zip(bars, vals):
    ax.text(v*1.6, b.get_y()+b.get_height()/2, f"{v:,}", va="center", fontsize=10)
ax.grid(axis="y", alpha=0)
fig.tight_layout(); fig.savefig("figs/params.pdf"); plt.close(fig)
print("figures written")
