import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('default')
# ---- Font settings (local machine) ----

matplotlib.rcParams['font.family'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
# ---- IRF parameters from IPCC AR6 ----

a = [0.2173, 0.2763, 0.2824, 0.224]
tau = [np.inf, 4.304, 36.54, 394.4]
# Parameters
dt = 0.1
tmax = 1000
t = np.arange(0, tmax + dt, dt)
# Compute each component

components = []
for i in range(4):
    if np.isinf(tau[i]):
        comp = np.full_like(t, a[i])
    else:
        comp = a[i] * np.exp(-t / tau[i])
    components.append(comp)

irf_total = sum(components)
# ---- Multiply by -1 for pCDR ----

components_neg = [-c for c in components]
irf_total_neg = -irf_total
# ---- Plot ----

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# Stack order: permanent(0), slow(3), medium(2), fast(1)

order = [0, 3, 2, 1]
stacked = [components_neg[i] for i in order]
# Nature-style muted palette

colors = ['#4a6274', '#c06c5a', '#e8b960', '#7db8a5']
labels = [
    r'$a_1$ = 0.2173 (permanent, $\tau_1 = \infty$)',
    r'$a_4$ = 0.2240, $\tau_4$ = 394.4 yr',
    r'$a_3$ = 0.2824, $\tau_3$ = 36.54 yr',
    r'$a_2$ = 0.2763, $\tau_2$ = 4.304 yr',
]

ax.stackplot(t, *stacked, colors=colors, labels=labels, alpha=0.9)
ax.plot(t, irf_total_neg, color='#1a1a1a', lw=1.5, label=r'Total IRF$_{\mathrm{CO_2}}$(t)')


ax.axvline(x=100, color='#aaaaaa', ls='--', lw=1)
idx_100 = np.argmin(np.abs(t - 100))
irf_100 = -sum(comp[idx_100] for comp in components)
ax.annotate(f't = 100 yr\nAtm. frac. $\\approx$ {irf_100:.2f}',
            xy=(100, irf_100), xytext=(170, -0.62),
            fontsize=12,
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8),
            ha='left')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Atmospheric fraction', fontsize=12)

ax.set_xlim(0, 1000)
ax.set_ylim(-1.05, 0)
ax.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='none')
ax.tick_params(labelsize=12)
# Clean up spines

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('figure/Figure Response pCDR.png', dpi=300, bbox_inches='tight', facecolor='white')
