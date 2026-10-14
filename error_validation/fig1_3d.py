import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(2); A = 4.0; r = 1.0
def unit(N, M):
    u = rng.standard_normal((M, N)); return u/np.linalg.norm(u, axis=1, keepdims=True)

fig = plt.figure(figsize=(15, 8.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.9, 1.25], height_ratios=[0.5, 1], wspace=0.15, hspace=0.35)

# ---------------- key: Eq. 5 -> shorthand ----------------
key = fig.add_subplot(gs[0, :]); key.axis('off')
key.text(0.0, 0.97, r"Paper Eq. 5:     $\mathbf{D_1}\tilde{\mathbf{i}}\;=\;\mathbf{D_1S}\tilde{\mathbf{i}}\;+\;\mathbf{D_1S}\boldsymbol{\Delta}$",
         fontsize=14, va='top', family='DejaVu Sans')
rows = [
    (r"$\mathbf{a}=\mathbf{D_1S}\tilde{\mathbf{i}}$", "signal vector: derivative of the (double-)smoothed profile; black arrow"),
    (r"$\mathbf{e}=\mathbf{D_1S}\boldsymbol{\Delta}$", "residual-noise vector: shifts the tip of $\\mathbf{a}$ onto the dashed circle / sphere"),
    (r"$\mathrm{SP}=\Vert\mathbf{a}+\mathbf{e}\Vert=\Vert\mathbf{D_1}\tilde{\mathbf{i}}\Vert$", "measured structure parameter (length of signal + noise)"),
    (r"$r=\Vert\mathbf{e}\Vert$", "length of the noise vector = radius of circle/sphere = SHRINE's error bar"),
    (r"$\mathrm{SP}-\Vert\mathbf{a}\Vert$", "how much the noise actually changed SP in one realisation (colour scale)"),
    (r"$N$", "number of independent noise directions (in SHRINE: $N_{\\rm eff}$, ~ modes below $k_c$)"),
]
for i, (sym, desc) in enumerate(rows):
    y = 0.50 - i*0.11
    key.text(0.0, y, sym, fontsize=11, va='top'); key.text(0.15, y, desc, fontsize=10.5, va='top')

# ---------------- (a) 2-D, as in paper Fig 1 ----------------
ax = fig.add_subplot(gs[1, 0]); u = unit(2, 400)
th = np.linspace(0, 2*np.pi, 200); ax.plot(A + r*np.cos(th), r*np.sin(th), '--', c='gray', lw=1)
ax.annotate('', xy=(A, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=2, color='k'))
tips = np.c_[A + r*u[:, 1], r*u[:, 0]]; L = np.linalg.norm(tips, axis=1)
sc = ax.scatter(tips[:, 0], tips[:, 1], c=L - A, cmap='coolwarm', vmin=-r, vmax=r, s=8, zorder=3)
k = int(np.argmin(abs(tips[:,1]+0.55*r)+abs(tips[:,0]-A-0.8*r)))
ax.annotate('', xy=tips[k], xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='C0'))
ax.annotate('', xy=tips[k], xytext=(A, 0), arrowprops=dict(arrowstyle='-|>', lw=1.2, color='C2'))
ax.text(A/2, 0.15, r'$\mathbf{a}=\mathbf{D_1S}\tilde{\mathbf{i}}$', ha='center', fontsize=11)
ax.text(A/2, tips[k, 1]/2 - 0.45, r'$\mathbf{a}+\mathbf{e}=\mathbf{D_1}\tilde{\mathbf{i}}$', color='C0', ha='center', fontsize=10)
mid = (np.array([A, 0]) + tips[k])/2
ax.text(mid[0] + 0.12, mid[1] - 0.05, r'$\mathbf{e}$', color='C2', fontsize=12)
ax.text(A + 1.1, 1.0, r'$r=\Vert\mathbf{e}\Vert=\Vert\mathbf{D_1S}\boldsymbol{\Delta}\Vert$', fontsize=10)
ax.set_aspect('equal'); ax.set_xlim(-0.3, A + 2.9); ax.set_ylim(-1.7, 1.7)
ax.set_title('N = 2 (as in paper Fig. 1)\neach dot = one possible tip of $\\mathbf{a}+\\mathbf{e}$')
ax.set_xticks([]); ax.set_yticks([])

# ---------------- (b) 3-D sphere ----------------
ax = fig.add_subplot(gs[1, 1], projection='3d'); u = unit(3, 1500)
tips = np.c_[r*u[:, 0], r*u[:, 1], A + r*u[:, 2]]; L = np.linalg.norm(tips, axis=1)
ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], c=L - A, cmap='coolwarm', vmin=-r, vmax=r, s=4, alpha=.8)
ax.quiver(0, 0, 0, 0, 0, A, color='k', lw=2, arrow_length_ratio=.08)
p = np.linspace(0, 2*np.pi, 100); ax.plot(r*np.cos(p), r*np.sin(p), A, c='k', lw=1.2)
ax.text(0.15, 0, A/2, r'$\mathbf{a}$', fontsize=12)
ax.set_box_aspect((1, 1, 2.2)); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(0, A + 1.2)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(18, 35)
ax.set_title('N = 3: tips of $\\mathbf{a}+\\mathbf{e}$ on a sphere of radius $r$\n'
             'poles: SP changes by $\\pm r$\nequator (black ring): no change', fontsize=10)

# ---------------- (c) distribution of SP - ||a|| vs N ----------------
ax = fig.add_subplot(gs[1, 2])
for N, c in [(2, 'C0'), (3, 'C1'), (40, 'C2'), (1000, 'C3')]:
    u = unit(N, 20000); d = np.sqrt(A**2 + 2*A*r*u[:, 0] + r**2) - A
    ax.hist(d, bins=np.linspace(-1.05, 1.05, 140), histtype='step', density=True, lw=1.5, color=c,
            label=f'N = {N}:  std = {d.std():.2f} r')
ax.axvspan(-r, r, color='gray', alpha=.12, label=r'SHRINE error bar: $\pm r$')
b = np.sqrt(A**2 + r**2) - A
ax.axvline(b, ls=':', c='k', lw=1); ax.text(b + 0.04, 7, r'bias $r^2/(2\Vert\mathbf{a}\Vert)$', fontsize=9)
ax.set_ylim(0, 12); ax.set_ylabel('fraction of noise realisations (density)')
ax.set_xlabel(r'SP $-\ \Vert\mathbf{a}\Vert$ : change in SP caused by noise  (units of $r$)')
ax.legend(fontsize=8.5, loc='upper left')
ax.set_title('Same noise length $r$, more dimensions:\nthe actual change in SP shrinks as $r/\\sqrt{N}$')

plt.colorbar(sc, ax=fig.axes[1:3], orientation='horizontal', shrink=.5, pad=.04,
             label=r'SP $-\ \Vert\mathbf{a}\Vert$ (units of $r$):  red = noise lengthens, blue = shortens')
plt.savefig('/home/joel/Documents/GitHub/SHRINE/error_validation/shrine_fig1_3d.png', dpi=300, bbox_inches='tight')
