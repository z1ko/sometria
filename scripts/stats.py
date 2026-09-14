from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import f as f_dist
from scipy.stats import shapiro
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# amass_clean / medium_100ep, 
# axes: input (p, pk, pkd) x loss (p, pk, pkd) x seed (1, 2, 3, 42)

CHOICES = ["p", "pk", "pkd"]
SEEDS = [1, 2, 3, 42]

macro_map = np.array([
    [  # input=p
        [0.347158, 0.344522, 0.339695, 0.347166],  # loss=p
        [0.360021, 0.366770, 0.360910, 0.361197],  # loss=pk
        [0.360968, 0.361034, 0.365398, 0.361589],  # loss=pkd
    ],
    [  # input=pk
        [0.352740, 0.355081, 0.349745, 0.344516],  # loss=p
        [0.364746, 0.365357, 0.369938, 0.363294],  # loss=pk
        [0.366315, 0.366704, 0.370877, 0.364404],  # loss=pkd
    ],
    [  # input=pkd
        [0.347914, 0.352773, 0.349112, 0.345188],  # loss=p
        [0.355125, 0.359587, 0.359298, 0.358531],  # loss=pk
        [0.360350, 0.363537, 0.366111, 0.357151],  # loss=pkd
    ],
])

micro_map = np.array([
    [  # input=p
        [0.529289, 0.529668, 0.528216, 0.534152],  # loss=p
        [0.547155, 0.555252, 0.551049, 0.551319],  # loss=pk
        [0.551416, 0.550187, 0.550811, 0.547844],  # loss=pkd
    ],
    [  # input=pk
        [0.546819, 0.543476, 0.540697, 0.539254],  # loss=p
        [0.550726, 0.554612, 0.553883, 0.553726],  # loss=pk
        [0.554715, 0.555604, 0.563801, 0.554323],  # loss=pkd
    ],
    [  # input=pkd
        [0.538467, 0.545154, 0.541112, 0.537563],  # loss=p
        [0.550616, 0.552392, 0.549877, 0.549583],  # loss=pk
        [0.550820, 0.550299, 0.551937, 0.549072],  # loss=pkd
    ],
])


# --------------------------------------------------------------------------- figures

# Tokens copied from scripts/label_analysis.py rather than imported -- that module pulls
# torch and the dataset layer in at import time. Same validated categorical slots 1-3;
# aqua sits under 3:1 on a light surface, so every figure using it carries direct labels.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e4e3de"

OUT = Path("images/stats")
INPUT_COLOUR = dict(zip(CHOICES, (BLUE, ORANGE, AQUA)))


def _style(axes, *, grid_axis: str = "y") -> None:
    """Recessive frame: no top/right spines, one faint grid axis, ink-token text."""

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID)
    axes.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    axes.set_axisbelow(True)



OUT = Path("images/stats")
INPUT_COLOUR = dict(zip(CHOICES, (BLUE, ORANGE, AQUA)))


def _style(axes, *, grid_axis: str = "y") -> None:
    """Recessive frame: no top/right spines, one faint grid axis, ink-token text."""

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID)
    axes.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    axes.set_axisbelow(True)


# --------------------------------------------------------------------------- analysis


def analyse(metric: str, data: np.ndarray, label: str) -> None:
    """The whole pipeline for one metric. Both metrics share the seeds and the
    design, so they get the same model -- only the numbers differ."""



    # Seeds are shared across all 9 cells, so input/loss are within-subject factors
    # and seed is the block. Treating the cells as independent groups pools the seed
    # variance into the error term and hides the interaction.

    long = pd.DataFrame([
        {"seed": SEEDS[k], "input": CHOICES[i], "loss": CHOICES[j],
         "comb": f"in_{CHOICES[i]}__loss_{CHOICES[j]}", "value": data[i, j, k]}
        for i in range(len(CHOICES)) for j in range(len(CHOICES)) for k in range(len(SEEDS))
    ])

    print(f"--- CELL MEANS ({metric}) ---")
    print(long.pivot_table(index="input", columns="loss", values="value",
                           aggfunc=["mean", "std"]).round(5))

    residuals = (data - data.mean(axis=2, keepdims=True)
                 - data.mean(axis=(0, 1), keepdims=True) + data.mean())

    print("\n--- REPEATED-MEASURES ANOVA ---")
    rm = AnovaRM(long, depvar="value", subject="seed", within=["input", "loss"]).fit()
    table = rm.anova_table
    # partial eta^2 from F and its df; with 4 seeds the p-values are fragile, so report effect size too
    table["part. eta^2"] = (table["F Value"] * table["Num DF"]) / (
        table["F Value"] * table["Num DF"] + table["Den DF"])

    # AnovaRM reports UNCORRECTED p-values, and sphericity does not hold here, which makes
    # them anti-conservative. Greenhouse-Geisser shrinks the df by epsilon. With 4 subjects
    # over 9 conditions the covariance is rank-deficient, so epsilon is itself badly
    # estimated -- the lower bound 1/df1 is reported as the worst case.
    Y = data.reshape(len(CHOICES) ** 2, len(SEEDS)).T
    unit = np.ones((len(CHOICES), 1)) / np.sqrt(len(CHOICES))
    within = np.linalg.qr(np.eye(len(CHOICES)) - unit @ unit.T)[0][:, :len(CHOICES) - 1]
    contrast_space = {"input": np.kron(within, unit), "loss": np.kron(unit, within),
                      "input:loss": np.kron(within, within)}

    for effect, basis in contrast_space.items():
        covariance = np.cov(Y @ basis, rowvar=False)
        df1 = basis.shape[1]
        eps = np.trace(covariance) ** 2 / (df1 * np.sum(covariance ** 2))
        f_value, df2 = table.loc[effect, "F Value"], table.loc[effect, "Den DF"]
        table.loc[effect, "eps_GG"] = eps
        table.loc[effect, "Pr > F (GG)"] = f_dist.sf(f_value, df1 * eps, df2 * eps)
        table.loc[effect, "Pr > F (lower)"] = f_dist.sf(f_value, df1 / df1, df2 / df1)

    print(table.round(4).to_string())
    print("\nSphericity is violated (eps < 1); read the GG column, not Pr > F.")
    print(f"Residual normality (Shapiro-Wilk): W={shapiro(residuals.ravel()).statistic:.3f}, "
          f"p={shapiro(residuals.ravel()).pvalue:.3f}")

    # Everything below supports one claim: the reconstruction target dominates the
    # encoder input, and the gain sits in a single step. ANOVA + marginals + three
    # contrasts, no cell-level reporting.

    SESOI = 0.005  # smallest effect worth caring about; set it from modelling impact, not data

    print("\n--- MARGINAL MEANS ---")
    for axis, name in [((1, 2), "input"), ((0, 2), "loss")]:
        means = data.mean(axis=axis)
        print(f"{name:6s} " + "  ".join(f"{c}={m:.4f}" for c, m in zip(CHOICES, means)))

    # Equal df, balanced design, so the sums of squares compare directly.
    grand = data.mean()
    ss = lambda m: len(SEEDS) * len(CHOICES) * ((m - grand) ** 2).sum()
    ss_input, ss_loss = ss(data.mean(axis=(1, 2))), ss(data.mean(axis=(0, 2)))
    print(f"\nSS loss = {ss_loss:.5f} vs SS input = {ss_input:.5f}  ({ss_loss / ss_input:.1f}x)")


    def contrast(name, a, b):
        """b - a over shared seeds: 95% CI, test against 0, and test against +/-SESOI."""
        d = b - a
        se = d.std(ddof=1) / np.sqrt(len(d))
        dof = len(d) - 1
        crit = t_dist.ppf(0.975, dof)
        return {"contrast": name, "diff": d.mean(),
                "ci_low": d.mean() - crit * se, "ci_high": d.mean() + crit * se,
                "p_raw": ttest_rel(b, a).pvalue,
                # a null result is not evidence of absence at n=4
                "p_tost": max(t_dist.sf((d.mean() + SESOI) / se, dof),
                              t_dist.cdf((d.mean() - SESOI) / se, dof))}


    marg = {(f, c): data.mean(axis=1 - f)[CHOICES.index(c)]
            for f in (0, 1) for c in CHOICES}  # f=0 input, f=1 loss
    FACTOR = {"input": 0, "loss": 1}
    STEPS = [("loss", "p", "pk"), ("loss", "pk", "pkd"), ("input", "pk", "pkd")]
    steps = [(f"{f:5s} {lo} -> {hi}", marg[FACTOR[f], lo], marg[FACTOR[f], hi])
             for f, lo, hi in STEPS]

    rows = [contrast(*s) for s in steps]
    reject, p_adj, *_ = multipletests([r["p_raw"] for r in rows], alpha=0.05, method="holm")
    res = pd.DataFrame(rows).assign(p_holm=p_adj, sig=reject)
    res["verdict"] = np.where(res.sig, "effect",
                              np.where(res.p_tost < 0.05, f"~0 (within {SESOI})", "undetermined"))
    print(f"\n--- MARGINAL CONTRASTS (Holm over {len(rows)}) ---")
    print(res[["contrast", "diff", "ci_low", "ci_high", "p_holm", "p_tost", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # The interaction does not survive the GG correction, but it is close enough that
    # the marginals still need checking: they are readable only if no step reverses sign
    # across the other factor's levels.
    print("\n--- SIGN CONSISTENCY (marginals readable only if all True) ---")
    for name, axis, lo, hi in [("loss  p -> pk ", 0, "p", "pk"), ("loss  pk -> pkd", 0, "pk", "pkd"),
                               ("input pk -> pkd", 1, "pk", "pkd")]:
        per = (data.take(CHOICES.index(hi), axis=1 - axis)
               - data.take(CHOICES.index(lo), axis=1 - axis)).mean(axis=1)
        print(f"{name}: per-level {np.array2string(per, precision=4, sign='+')}  "
              f"same sign = {bool(np.all(np.sign(per) == np.sign(per.sum())))}")



    def plot_interaction():
        """Loss on x, one line per input level. The rise across x against the spread
        between lines is the 9.5x SS ratio, made visual."""

        figure, axes = plt.subplots(figsize=(6.4, 4.0))
        x = np.arange(len(CHOICES))

        # `input p` and `input pkd` land within 0.0005 of each other, so the direct
        # labels need nudging apart or they overprint.
        ends = sorted((data[i, -1].mean(), i) for i in range(len(CHOICES)))
        gap = (data.max() - data.min()) * 0.06
        label_y = {}
        for k, (value, i) in enumerate(ends):
            floor = label_y[ends[k - 1][1]] + gap if k else -np.inf
            label_y[i] = max(value, floor)

        for i, level in enumerate(CHOICES):
            colour = INPUT_COLOUR[level]
            cells = data[i]                       # (loss, seed)
            # Seeds first, faint, so the reader sees n=4 rather than a mean with no data.
            for j in x:
                axes.scatter(np.full(len(SEEDS), j) + (i - 1) * 0.055, cells[j],
                             s=14, color=colour, alpha=0.35, lw=0, zorder=2)
            axes.plot(x, cells.mean(axis=1), color=colour, lw=2, marker="o", ms=6,
                      mec="white", mew=1.5, zorder=3)
            # Direct label: identity is never colour-alone, and it is the relief the
            # aqua contrast WARN requires.
            axes.annotate(f"input {level}", (x[-1] + 0.08, label_y[i]),
                          color=colour, fontsize=9, weight="bold", va="center")

        _style(axes)
        axes.set_xticks(x, [f"loss {c}" for c in CHOICES])
        axes.set_xlim(-0.3, len(CHOICES) + 0.15)
        axes.set_ylabel(label, color=INK_2, fontsize=9)
        axes.set_title("Reconstruction target drives the probe; encoder input barely moves it",
                       color=INK, fontsize=10, loc="left", pad=12)
        figure.tight_layout()
        return figure


    def plot_slopes():
        """One panel per reported contrast, one line per seed. Shows the pairing that
        the repeated-measures model relies on."""

        # sharey: on independent axes the +0.0016 step reads as large as the +0.0141 one.
        figure, panels = plt.subplots(1, len(steps), figsize=(7.6, 3.2), sharey=True)
        for axes, (factor, lo, hi), (_, a, b) in zip(panels, STEPS, steps):
            for y0, y1 in zip(a, b):
                axes.plot([0, 1], [y0, y1], color=BLUE, lw=1.4, alpha=0.55,
                          marker="o", ms=5, mec="white", mew=1.2)
            axes.plot([0, 1], [a.mean(), b.mean()], color=INK, lw=2.4, zorder=4)
            _style(axes)
            axes.set_xticks([0, 1], [lo, hi])
            axes.set_xlim(-0.25, 1.25)
            axes.set_title(f"{factor} {lo} -> {hi}   ({b.mean() - a.mean():+.4f})",
                           color=INK_2, fontsize=9, loc="left")
        panels[0].set_ylabel(f"{label} (marginal)", color=INK_2, fontsize=9)
        figure.suptitle(f"One line per seed, shared y-axis (n={len(SEEDS)})",
                        color=INK, fontsize=10, x=0.01, ha="left")
        figure.tight_layout()
        return figure


    def plot_contrasts():
        """Effect + 95% CI against zero and the equivalence band -- the verdict column,
        drawn. A CI inside the band is the TOST result."""

        figure, axes = plt.subplots(figsize=(6.8, 2.6))
        y = np.arange(len(res))[::-1]
        axes.axvspan(-SESOI, SESOI, color=GRID, alpha=0.7, zorder=0)
        axes.axvline(0, color=INK_3, lw=1, zorder=1)
        axes.hlines(y, res.ci_low, res.ci_high, color=BLUE, lw=2.2, zorder=3)
        axes.scatter(res["diff"], y, s=46, color=BLUE, zorder=4)

        for yi, (_, row) in zip(y, res.iterrows()):
            axes.annotate(row.verdict, (row.ci_high, yi), xytext=(10, 0),
                          textcoords="offset points", color=INK_2, fontsize=8, va="center")
        _style(axes, grid_axis="x")
        span = res.ci_high.max() - res.ci_low.min()
        axes.set_xlim(res.ci_low.min() - 0.1 * span, res.ci_high.max() + 0.45 * span)
        axes.set_yticks(y, [c.strip() for c in res.contrast])
        axes.set_xlabel(f"change in {label} (shaded = equivalence band, +/-{SESOI})",
                        color=INK_2, fontsize=9)
        axes.set_title("Effect sizes with 95% CI", color=INK, fontsize=10, loc="left", pad=10)
        figure.tight_layout()
        return figure



    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in [("interaction", plot_interaction),
                        ("slopes", plot_slopes),
                        ("contrasts", plot_contrasts)]:
        build().savefig(OUT / f"{metric}_{name}.png", dpi=160, facecolor="#fcfcfb")
        print(f"wrote {OUT / f'{metric}_{name}.png'}")


for metric, data, label in [("macro_map", macro_map, "macro mAP"),
                            ("micro_map", micro_map, "micro mAP")]:
    print("\n" + "=" * 78 + f"\n{metric}\n" + "=" * 78)
    analyse(metric, data, label)
