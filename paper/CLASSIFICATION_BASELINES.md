# Reference Guide: What a Probe Score Is Worth

**Repository Reference Document**
*Topic:* Baselines & Readout Budget for Frozen-Backbone Evaluation
*Context:* BABEL-60 window multilabel classification
*Supersedes:* Section 6 of `CHANNEL_ABLATION.md` ("A statistical baseline beats us")

---

## Key Quantitative Summary

| Question | Weak Readout | Strong Readout | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Does pretraining beat no pretraining?** | $2.75\times$ | $1.97\times$ | The headline ratio moves $28\%$ **without touching the representation** |
| **Does pretraining beat summary statistics?** | $1.07\times$ | $1.15\times$ | A pretrained transformer beats 860 hand-computed numbers by $7$–$15\%$ |
| **Who benefits from a better head?** | --- | random $+0.062$ vs pretrained $+0.024$ | Extra readout capacity is worth **more** on an *untrained* encoder |

**Bottom line:** window classification cannot separate "the representation is good" from
"the task is easy." Use it to report absolute numbers, not to compare methods.

---

## 1. The Three Inputs

All three feed the *same* multilabel head, on the same windows, with the same optimizer.
Only what enters the head changes.

| Input | What it is | Learned params in the encoder |
| :--- | :--- | :---: |
| **Pretrained encoder** | MAE, mask ratio $0.9$, depth 8, $d_{\text{model}}=256$, 40 epochs on `pretrain_v1/train` | 6.3 M (frozen) |
| **Moments** | mean, std, min, max per DoF per channel $= 43 \times 5 \times 4 = 860$ numbers | 0 |
| **Random encoder** | identical architecture, random init | 6.3 M (frozen, untrained) |

The random encoder says *how much of the score is due to pretraining*.
The moments say *how much of the score was available without learning anything*.
Neither alone bounds the result.

---

## 2. Readout Budget Is Not an Implementation Detail

Two probe budgets over identical frozen backbones:
**weak** = $\text{lr}=10^{-3}$, 20 epochs. **strong** = $\text{lr}=5\times10^{-3}$, 40 epochs.
Both use `ATTENTIVE` pooling.

### Table 1: Same Backbones, Two Heads

| Input to the head | Macro mAP (weak) | Macro mAP (strong) | $\Delta$ | Relative |
| :--- | :---: | :---: | :---: | :---: |
| Pretrained encoder | 0.351 | **0.375** | $+0.024$ | $+6.8\%$ |
| Moments | 0.328 | 0.327 | $-0.002$ | $-0.6\%$ |
| Random encoder | 0.128 | 0.190 | $+0.062$ | $\mathbf{+48.7\%}$ |
| | | | | |
| *pretrained / random* | *2.75* | *1.97* | | |
| *pretrained / moments* | *1.07* | *1.15* | | |

### Key Takeaways:
1. **The control moves more than the treatment.** A random encoder gains $+0.062$ macro
   mAP from a better-tuned head — two and a half times what the pretrained encoder gains,
   obtained with zero change to the representation.
2. **The headline ratio is a function of probe tuning.** $2.75\times$ and $1.97\times$
   describe the same backbone. Any paper reporting "$N\times$ over random init" without
   its probe budget is reporting an uncalibrated number.
3. **Moments are invariant.** A fixed 860-vector into a linear layer has no pooling to
   learn, so it cannot absorb head capacity. That is what makes it the **stable**
   reference of the two, and why the pretrained/moments ratio *rises* while
   pretrained/random *falls*.
4. **Mechanism.** Attentive pooling over a $T \times D$ grid can learn *which tokens to
   read* even when the token contents are random projections. Much of what a probe
   credits to the representation is the readout finding structure in the input the
   encoder merely passed through.

---

## 3. The Metrics

All numbers are computed on `babel_official/val`: **4936 windows**, a **60-label**
vocabulary, **all 60 labels present**, mean **2.525** labels per window (median 2). A
label counts for a window when it covers at least `label_min_coverage` of it (0.15 for
classification, 0.5 for segmentation).

Two of these metrics are threshold-free and two are not. That distinction decides which
ones may be compared across experiments.

### 3.1 Macro mAP — *the headline metric*

`WindowMeanAveragePrecision` (`downstream/metrics.py`). For each label, rank all 4936
windows by that label's predicted score and compute average precision; then average over
labels. **Every label counts equally**, so the 60th-most-common action matters as much as
walking.

One deviation from the torchmetrics default: labels with **no positive** in the targets
are excluded from the average rather than scored 0. With a frequency-ordered vocabulary,
including them would report *how much of the vocabulary happens to occur* instead of model
quality. On this split all 60 labels occur, so the filter is currently inert — it matters
if the vocabulary is widened.

* **Threshold-free.** It scores the *ranking*, so it is unaffected by calibration.
* **Chance = 0.0421** (mean label prevalence).
* **Comparable across everything** — readout budgets, tasks, papers. This is why it is
  the metric every claim in this document rests on.

### 3.2 Micro mAP

`MultilabelAveragePrecision(average="micro")`. Pools all $4936 \times 60$ window-label
pairs into a **single** ranking and computes one average precision over it.

* Frequency-weighted by construction: the most common label appears in **39.9%** of
  windows, the rarest present one in **0.49%**, so micro mAP is dominated by a handful of
  head classes.
* **Chance = 0.0421** (overall positive rate).
* Useful as a sanity check, weak as a discriminator — it compresses the very differences
  macro mAP is designed to expose. Compare `0.550 / 0.507 / 0.373` (micro) against
  `0.375 / 0.327 / 0.190` (macro) for the same three inputs.

### 3.3 Macro F1 — *threshold-dependent, handle with care*

`MultilabelF1Score(threshold=0.5, average="macro")`. Binarizes each score at **0.5**,
computes per-label F1, averages over labels.

* **Not threshold-free.** The median window carries 2 of 60 labels, so a well-ranked but
  conservatively-calibrated model can score near zero while its mAP is high.
* This is why macro F1 jumps $0.2226 \to 0.3467$ ($+56\%$) between readout budgets where
  macro mAP moves $+6.8\%$: the higher learning rate pushes more logits past 0.5 without
  ranking anything better.
* **Do not compare macro F1 across experiments that differ in optimizer settings.** Within
  one fixed budget it is informative about the rare classes; across budgets it is an
  artifact of calibration.

### 3.4 Recall@$k$ — R@1, R@3, R@5

`MultilabelTopKRecall`. Take the $k$ highest-scoring labels for a window; count how many
of that window's true labels are among them; sum over windows and divide by the total
number of positives.

**Recall, not precision, and capped below 1.** Top-$k$ can retrieve at most $k$ labels, so
a window with 3 labels scores at most $1/3$ at $k=1$. Ceilings on this split:

| | R@1 | R@3 | R@5 |
| :--- | :---: | :---: | :---: |
| **Maximum achievable** | 0.3792 | 0.8170 | 0.9593 |

Always quote the ceiling beside the value, or a legitimate 0.284 reads as failure.

### 3.5 Boundary F1 — segmentation only

`boundary_f1` (`downstream/segmentation.py`). A *boundary* is a patch index where the
multi-hot target differs from the previous patch. Binarize predictions at a threshold,
find where they change, and take F1 between predicted and actual change positions.

* Scores **only where the target moves**. A prediction constant across the window earns
  exactly **0.000**, however well it names the action — which is what makes it the one
  metric a window-level summary statistic cannot fake.
* `boundary_best_f1s` is the same quantity **maximized over 30 thresholds**
  ($0.02$ to $0.60$ in steps of $0.02$) chosen on the validation set. It is therefore
  **optimistically biased and not a held-out number**: random per-patch scores reach
  **0.148** under the sweep against a boundary base rate of **0.080**. Report the
  fixed-threshold version; the swept one exists to show how much of a boundary score is
  threshold fitting.

### 3.6 Which metric to trust

| Metric | Threshold-free | Class-balanced | Comparable across budgets |
| :--- | :---: | :---: | :---: |
| **Macro mAP** | yes | yes | **yes** |
| Micro mAP | yes | no | yes |
| Macro F1 | **no** | yes | **no** |
| Recall@$k$ | yes | no | yes (with ceiling) |
| Boundary F1 | **no** | --- | within a threshold |
| Boundary F1 (swept) | --- | --- | **no** (fitted on val) |

---

## 4. Full Metrics

### Table 2: Window Classification, All Metrics

| Input | Readout | Macro mAP | Micro mAP | Macro F1 | R@1 | R@3 | R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pretrained (s13) | weak | 0.3507 | 0.5378 | 0.2226 | 0.2797 | 0.5265 | 0.6413 |
| Pretrained (s14) | weak | 0.3498 | 0.5403 | 0.2352 | 0.2808 | 0.5272 | 0.6443 |
| Moments | weak | 0.3284 | 0.5112 | 0.2431 | 0.2695 | 0.5121 | 0.6250 |
| Random | weak | 0.1276 | 0.3341 | 0.0213 | 0.2076 | 0.3733 | 0.4701 |
| **Pretrained (s13)** | **strong** | **0.3746** | **0.5504** | **0.3467** | **0.2841** | **0.5320** | **0.6450** |
| Moments | strong | 0.3265 | 0.5074 | 0.2746 | 0.2717 | 0.5086 | 0.6208 |
| Random | strong | 0.1898 | 0.3726 | 0.0789 | 0.2178 | 0.4026 | 0.5035 |

*Seed spread (attentive, weak, two seeds): $0.0009$ macro mAP. No second seed exists for
the strong readout.*

### Key Takeaways:
1. **Ranking the full label set separates; retrieving the top one does not.** Against the
   ceilings in §3.4, R@1 is $74.9\%$ (pretrained) vs $71.7\%$ (moments) vs $57.4\%$
   (random) — a $3.2$-point gap where macro mAP shows $15\%$. Naming the single most
   likely action is largely a summary-statistics problem.
2. **Ignore the macro F1 column across rows of different readout.** See §3.3: the
   $+56\%$ jump is calibration crossing the $0.5$ threshold, not better ranking. Within a
   budget it is informative; between budgets it is an artifact.
3. **Micro mAP compresses everything.** $0.550 / 0.507 / 0.373$ against macro's
   $0.375 / 0.327 / 0.190$ for the same three inputs — the head classes the moments
   already solve dominate it.
4. **The moments reach $87\%$ of the pretrained encoder** on macro mAP under the strong
   readout ($0.3265$ vs $0.3746$), and $94\%$ of it under the weak one. That range is the
   discriminative capacity of this benchmark.

--- | :---: | :---: | :---: |
   | Pretrained | 74.9% | 65.1% | 67.2% |
   | Moments | 71.7% | 62.3% | 64.7% |
   | Random | 57.4% | 49.3% | 52.5% |

   The pretrained/moments gap is $3.2$ points of ceiling at $k=1$ against $15\%$ on macro
   mAP: retrieving the *single* most likely action is largely a summary-statistics
   problem, while ranking the full label set is where the representation shows up.
2. **Macro F1 is not comparable across readout budgets.** It jumps $0.2226 \to 0.3467$
   ($+56\%$) for the pretrained encoder where macro mAP moves $+6.8\%$. `MultilabelF1Score`
   thresholds at $0.5$ and the median window carries 2 of 60 labels, so a
   higher-learning-rate head crossing $0.5$ more often inflates F1 without ranking anything
   better. **Do not quote F1 ratios between the two budgets.** mAP is threshold-free and
   unaffected.
3. **Micro mAP compresses everything.** $0.55$ vs $0.51$ vs $0.37$: frequency-weighted
   metrics are dominated by the head classes, which the moments already solve.

---

## 5. What This Means

* **The channel-ablation conclusions are unaffected.** Every arm in `CHANNEL_ABLATION.md`
  was measured at one fixed readout budget, so the *ordering* stands. What changes is the
  absolute framing: "pretraining yields a transferable representation" is a weak claim when
  860 arithmetic features reach $87\%$ of it.
* **Report both baselines, always.** The random encoder alone overstates by a factor that
  depends on how hard you tuned the probe.
* **Window classification is the wrong benchmark for method comparison.** Its
  discriminative range is roughly $0.33 \to 0.37$ between "no learning at all" and "40
  epochs of pretraining." Use a task whose target varies within the window; see
  `baselines_and_segmentation.tex` §2, where the same backbones separate $2.70\times$ on
  per-patch prediction and the ratio *survives* the strong-readout control.

---

## 6. Provenance

Every number above is extracted from a stored run or a captured log — see
`paper/data/PROVENANCE.md`. Regenerate the run-derived values with:

    uv run python paper/data/extract.py <run dirs>

Superseded values previously circulated: moments weak $0.3299 \to$ **0.3284** (regenerated
at the matching budget); all "random init" numbers must now carry a readout-budget label.
