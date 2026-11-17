"""Build the report figures from SAVED experimental results only.

Nothing here changes the experiments or the saved results. Every figure is computed from the files in results/.
One exception, for Figure 1: the experiment saved the final states but not the corrupted inputs. To show the
real corrupted cue, this script replays the saved random-number stream of the two example groups, using the
original seeds and the unchanged src/hopfield.py. It then checks that EVERY replayed final state is
bit-identical to the saved final state, and stops with an error if any differs. So the cue shown is exactly
the one used in the experiment; no new result is produced.

Run from the project root:  python scripts/make_report_figures.py
Outputs: results/figures/report_fig*.png and results/figures/appendix/*.png
"""
import base64
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy.special import expit

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.optdigits import load_optdigits
from src.hopfield import train_hebbian, recall, corrupt, overlap   # unchanged implementation

R = ROOT / "results"
FIG = R / "figures"
APP = FIG / "appendix"
APP.mkdir(parents=True, exist_ok=True)

images, labels = load_optdigits()
N = 1024
patterns = 2.0 * images.reshape(len(images), N) - 1.0
cfg5 = json.loads((R / "step5_config.json").read_text())
edges = np.array(cfg5["BIN_EDGES"])
FIG_RNG = np.random.default_rng(12345)          # only for bootstrap CIs drawn in figures
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def boot_ci(x, B=2000):
    x = np.asarray(x, dtype=float)
    m = FIG_RNG.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


groups5 = pd.read_csv(R / "step5_groups.csv")
summary5 = pd.read_csv(R / "step5_group_summary.csv")
trials5 = pd.read_csv(R / "step5_trials.csv")
states5 = np.load(R / "step5_final_states.npy")
gimgs = {r.group_id: np.array(list(map(int, r.images.split()))) for r in groups5.itertuples()}


def unpack(b):
    return np.unpackbits(b)[:N].astype(np.float64) * 2 - 1


# ============================================================ Figure 1: retrieval examples (replayed, verified)
def replay_group(g):
    """Re-run group g's trials with its original seed stream. Return the cues and check every final state."""
    seeds = np.random.SeedSequence(cfg5["TRIAL_SEED"]).spawn(len(groups5))
    rng = np.random.default_rng(seeds[g])
    stored = patterns[gimgs[g]]
    W = train_hebbian(stored)
    rows = trials5[trials5.group_id == g].sort_values("trial_id")
    cues = {}
    it = iter(rows.itertuples())
    for level in cfg5["CORRUPTION_LEVELS"]:
        for pos in range(3):
            for t in range(cfg5["TRIALS_PER_IMAGE"]):
                r = next(it)
                assert (r.corruption, r.image_pos, r.trial) == (level, pos, t)
                cue = corrupt(stored[pos], level, rng)
                res = recall(W, cue, rng, max_sweeps=cfg5["MAX_SWEEPS"])
                assert np.array_equal(res["state"], unpack(states5[r.trial_id])), f"replay mismatch at trial {r.trial_id}"
                cues[r.trial_id] = cue
    return cues


def pick_trial(g, level, pos, want_correct):
    t = trials5[(trials5.group_id == g) & (trials5.corruption == level) & (trials5.image_pos == pos)]
    t = t[(t.outcome == "correct") == want_correct]
    return int(t.trial_id.iloc[0])


lo_g = int(summary5.sort_values("mean_overlap").iloc[0].group_id)                   # lowest-overlap group
hi_g = int(summary5[summary5.all_merged_20].sort_values("mean_overlap").iloc[-1].group_id)  # highest, fully merged
LEVEL = 0.20
examples = [("Retrieval succeeds", lo_g, pick_trial(lo_g, LEVEL, 0, True)),
            ("Retrieval fails: majority-vote mixture", hi_g, pick_trial(hi_g, LEVEL, 0, False))]

fig, axes = plt.subplots(2, 6, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.25, 1, 1]})
for row, (title, g, tid) in enumerate(examples):
    cues = replay_group(g)                                                       # verified replay
    stored = patterns[gimgs[g]]
    final = unpack(states5[tid])
    M = np.sign(stored.sum(axis=0))
    tr = trials5.loc[tid]
    for k in range(3):
        ax = axes[row, k]
        ax.imshow(stored[k].reshape(32, 32), cmap="gray_r", vmin=-1, vmax=1)
        ax.set_title(f"stored {k+1}: digit {labels[gimgs[g][k]]}" + ("  (target)" if k == tr.image_pos else ""), fontsize=9)
        if k == tr.image_pos:
            for s in ax.spines.values():
                s.set_edgecolor("tab:blue"); s.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
    axes[row, 3].axis("off")
    axes[row, 3].text(0.5, 0.5, "→", fontsize=26, ha="center", va="center")
    ax = axes[row, 4]
    ax.imshow(cues[tid].reshape(32, 32), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_title(f"cue: target, {int(LEVEL*100)}% pixels flipped", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[row, 5]
    ax.imshow(final.reshape(32, 32), cmap="gray_r", vmin=-1, vmax=1)
    same_M = np.array_equal(final, M)
    label = "= target exactly" if tr.outcome == "correct" else ("= majority vote M exactly" if same_M else tr.outcome)
    ax.set_title(f"final state {label}\nm(final, target) = {overlap(final, stored[tr.image_pos]):.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    axes[row, 0].set_ylabel(f"{title}\n(group mean overlap {summary5.loc[summary5.group_id == g, 'mean_overlap'].item():.2f})",
                            fontsize=9)
fig.suptitle("Three-memory networks: one saved trial from a low-overlap group and one from a high-overlap group", y=1.01)
fig.text(0.5, -0.02, "Stored images and final states are from the saved results (step5_final_states.npy). Cues come from a "
         "replay of the original seed stream; every replayed final state matched the saved state exactly.",
         ha="center", fontsize=8)
save(fig, "report_fig1_retrieval_examples")

# ============================================================ Figure 2: memory load (supporting evidence)
pilot = pd.read_csv(R / "pilot_groupsize_groups.csv")
s4 = pd.read_csv(R / "step4_groups.csv")
sets = [
    ("k=2\npilot", 2, pilot[pilot["size"] == 2].n_stable / 2, pilot[pilot["size"] == 2].mean_overlap, "tab:blue", "o"),
    ("k=3\npilot", 3, pilot[pilot["size"] == 3].n_stable / 3, pilot[pilot["size"] == 3].mean_overlap, "tab:orange", "o"),
    ("k=3\nExp. 3", 3, groups5.n_stable / 3, groups5.mean_overlap, "tab:orange", "s"),
    ("k=4\npilot", 4, pilot[pilot["size"] == 4].n_stable / 4, pilot[pilot["size"] == 4].mean_overlap, "tab:green", "o"),
    ("k=5\nExp. 1", 5, s4.n_stable / 5, s4.mean_overlap, "tab:red", "s"),
]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
for x, (name, k, st, mo, col, mk) in enumerate(sets):
    lo, hi = boot_ci(st)
    ax.errorbar(x, st.mean(), yerr=[[st.mean() - lo], [hi - st.mean()]], marker=mk, color=col, ms=8, capsize=4)
    ax.text(x, -0.13, f"n = {len(st)}", ha="center", fontsize=8)
ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[0] for s in sets])
ax.set_ylim(-0.18, 1.08); ax.set_ylabel("mean clean stability\n(fraction of stored images that are exact fixed points)")
ax.set_title("A. Clean stability by number of stored images")
ax.grid(alpha=0.3, axis="y")
ax = axes[1]
for name, k, st, mo, col, mk in sets:
    jit = FIG_RNG.uniform(-0.02, 0.02, len(st))
    ax.scatter(mo, st + jit, s=16 if mk == "o" else 9, color=col, marker=mk, alpha=0.6, label=name.replace("\n", " "))
ax.set_xlabel("group mean pairwise overlap"); ax.set_ylabel("clean stability of group")
ax.set_title("B. Group-level values (each point = one group)")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.text(0.5, -0.06,
         "Supporting evidence, not a controlled comparison. The groups were sampled differently: pilot = 4 groups per percentile-defined "
         "overlap level per size;\nExp. 1 = stratified by absolute overlap (7 bins x 12); Exp. 3 = stratified, all possible groups listed "
         "(15 bins, 180 groups). Overlap ranges therefore differ between sets. Error bars: bootstrap 95% CI over groups.",
         ha="center", fontsize=8)
save(fig, "report_fig2_memory_load")

# ============================================================ Figure 3: overlap vs stability and retrieval
stats = pd.read_csv(R / "step5_statistics.csv")
cols = [("stable_frac", "0% (clean stability)", "k"), ("acc_5", "5% corruption", "tab:blue"),
        ("acc_10", "10% corruption", "tab:orange"), ("acc_20", "20% corruption", "tab:red")]
names = ["clean stability (primary)", "accuracy 5%", "accuracy 10%", "accuracy 20%"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
ax = axes[0]
ax.axvspan(0.44, 0.56, color="0.92", zorder=0)
xs = np.linspace(summary5.mean_overlap.min(), summary5.mean_overlap.max(), 400)
for (col, lab, c), nm in zip(cols, names):
    bx, by, blo, bhi = [], [], [], []
    for b in sorted(summary5.bin.unique()):
        d = summary5[summary5.bin == b]
        bx.append(d.mean_overlap.mean()); by.append(d[col].mean())
        l, h = boot_ci(d[col]); blo.append(by[-1] - l); bhi.append(h - by[-1])
    ax.errorbar(bx, by, yerr=[blo, bhi], color=c, marker="o", ms=3.5, lw=0, elinewidth=1, capsize=2, label=lab)
    s = stats[stats.outcome == nm].iloc[0]
    ax.plot(xs, expit(-s.slope * (xs - s.m50)), color=c, lw=1.5)
ax.set_xlabel("group mean pairwise overlap (raw bipolar pixel overlap)")
ax.set_ylabel("group-level fraction (bin mean ± 95% CI)")
ax.set_title("A. Three memories: stability and exact retrieval vs. overlap (180 groups)")
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
ax.text(0.50, 1.02, "denser sampling", ha="center", fontsize=8, color="0.4")
ax = axes[1]
lv = [0, 5, 10, 20]
m50 = [stats[stats.outcome == nm].m50.item() for nm in names]
lo = [stats[stats.outcome == nm].m50_ci_low.item() for nm in names]
hi = [stats[stats.outcome == nm].m50_ci_high.item() for nm in names]
ax.errorbar(lv, m50, yerr=[np.subtract(m50, lo), np.subtract(hi, m50)], marker="o", color="k", capsize=4)
for x, y in zip(lv, m50):
    ax.text(x + 0.6, y + 0.001, f"{y:.3f}", fontsize=8)
ax.set_xlabel("corruption level (%)"); ax.set_ylabel("overlap at 50% (descriptive logistic fit)")
ax.set_title("B. More noise: failure at lower overlap"); ax.set_xticks(lv); ax.grid(alpha=0.3)
fig.text(0.5, -0.13, "Curves: descriptive logistic fits to group-level fractions, fitted only after pre-set monotonicity and "
         "50%-crossing checks.\nThe 50% points describe this sample (typical images, 3 different digits per group), not a universal "
         "threshold.\nCIs treat groups as independent and are optimistic because some images appear in several groups.",
         ha="center", fontsize=8)
save(fig, "report_fig3_overlap_retrieval")

# ============================================================ Figure 4: stability analysis
mech = pd.read_csv(R / "step5_mechanism_checks.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
for st, c, lab in [(True, "tab:green", "stable (exact fixed point)"), (False, "tab:red", "unstable")]:
    d = mech[mech.stable_observed == st]
    ax.scatter(d.mean_overlap, d.pair_sum, s=9, alpha=0.6, color=c, label=f"{lab}, n = {len(d)}")
xx = np.array([0.2, 0.76])
ax.plot(xx, 2 * xx, color="0.4", ls=":", lw=1, label="2 x group mean (average of the pair sums)")
ax.axhline(1 - 3 / N, color="k", ls="--", lw=1.2, label="rule boundary: 1 - 3/N")
ax.set_xlabel("group mean pairwise overlap"); ax.set_ylabel(r"$m_{ab}+m_{ac}$ (image's overlaps with the other two)")
ax.set_title("A. Stability rule: stable iff $m_{ab}+m_{ac} < 1-3/N$\n(rule matches 540/540 images)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
ax.text(0.98, 0.03, "Valid under assumptions A1-A4\n(checked for all 540 images):\nA1 >=1 minority pixel; A2 |m_ab - m_ac| < 1-3/N;\n"
        "A3 m_ab + m_ac > -(1-3/N); A4 N even (no ties)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.7"))
ax = axes[1]
st = mech[mech.stable_observed].copy()
bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
st["mb"] = pd.cut(st.margin, bins)
tab = st.groupby("mb", observed=True)[["acc_5", "acc_10", "acc_20"]].mean()
cnt = st.groupby("mb", observed=True).size()
x = np.arange(len(tab))
for j, (c, colr, lab) in enumerate([("acc_5", "tab:blue", "5%"), ("acc_10", "tab:orange", "10%"), ("acc_20", "tab:red", "20%")]):
    ax.bar(x + (j - 1) * 0.27, tab[c], width=0.27, color=colr, label=f"{lab} corruption")
ax.set_xticks(x)
ax.set_xticklabels([f"{iv.left:.2f}-{iv.right:.2f}\n(n={n})" for iv, n in zip(tab.index, cnt)], fontsize=8)
ax.set_xlabel(r"stability margin $1-3/N-(m_{ab}+m_{ac})$ of stable images")
ax.set_ylabel("exact retrieval accuracy (image level)")
ax.set_title("B. Barely stable images fail under corruption\n(unstable images: 0 accuracy at every level)")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False); ax.grid(alpha=0.3, axis="y")
fig.text(0.5, -0.12, "A: mathematical deduction, checked against all 540 stored images (observation). "
         "B: observation; explaining it by noise around a small margin is an interpretation, not a derivation.",
         ha="center", fontsize=8)
save(fig, "report_fig4_stability_analysis")

# ============================================================ Figure 5: overlap decomposition
pairs = pd.read_csv(R / "step5_overlap_decomposition.csv")
grp = pairs.groupby("group_id").mean(numeric_only=True)
dec = grp.groupby("bin")[["m", "p00_border", "p00_central", "p11", "p_dis"]].mean()
blabels = [f"{edges[int(b)]:.2f}-{edges[int(b)+1]:.2f}" for b in dec.index]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
ax = axes[0]
x = np.arange(len(dec))
ax.bar(x, dec.p00_border, color="0.72", label="both background: border pixels (ink in <5% of typical images)")
ax.bar(x, dec.p00_central, bottom=dec.p00_border, color="0.9", edgecolor="0.6", lw=0.5, label="both background: other pixels")
ax.bar(x, dec.p11, bottom=dec.p00_border + dec.p00_central, color="k", label="both ink")
ax.bar(x, -dec.p_dis, color="tab:red", alpha=0.75, label="disagree (counts negatively)")
ax.plot(x, dec.m, "o-", color="tab:blue", lw=2, label="bipolar overlap m = sum of the parts")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(blabels, rotation=55, ha="right", fontsize=8)
ax.set_xlabel("overlap bin (Exp. 3)"); ax.set_ylabel("contribution to m (group mean)")
ax.set_title(r"A. $m = p_{00} + p_{11} - p_{dis}$: what makes up the overlap")
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.32), ncol=2, frameon=False)
ax = axes[1]
sc = ax.scatter(pairs.m, pairs.ink_jaccard, c=(pairs.ink_a + pairs.ink_b) / 2, cmap="viridis", s=9)
plt.colorbar(sc, ax=ax, label="mean ink fraction of the pair")
ax.set_xlabel("bipolar overlap m"); ax.set_ylabel("shared ink / ink in either image")
ax.set_title("B. Overlap ranks pairs like ink-only shape overlap\n(Spearman rho = 0.965, 540 pairs)")
ax.grid(alpha=0.3)
save(fig, "report_fig5_overlap_decomposition")

# ============================================================ Appendix: images already embedded in executed notebooks
for nb_name in ["01_dataset_exploration", "02_patterns_and_similarity", "03_hopfield_verification", "04_similarity_experiment"]:
    nb = nbformat.read(ROOT / "notebooks" / f"{nb_name}.ipynb", as_version=4)
    k = 0
    for ci, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                k += 1
                p = APP / f"{nb_name}_cell{ci:02d}_{k}.png"
                p.write_bytes(base64.b64decode(data["image/png"]))
    print(f"{nb_name}: extracted {k} saved plot(s) to {APP}")
