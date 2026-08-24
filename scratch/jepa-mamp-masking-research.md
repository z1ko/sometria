# JEPA, MAE, and MAMP masking semantics

Date: 2026-08-24

## Question

Does `mask_ratio` mean the target/masked fraction or the context/visible fraction in upstream MAMP and Sometria? Does it make sense that Sometria uses high ratios for MAE/MAMP but a lower ratio for JEPA? Is motion-aware masking conceptually correct for JEPA?

## Primary sources checked

- Upstream MAMP repository: <https://github.com/maoyunyao/MAMP>
- Upstream MAMP MAMP transformer: <https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py>
- Upstream MAMP MAE transformer: <https://github.com/maoyunyao/MAMP/blob/main/model_mae/transformer.py>
- MAMP paper PDF: <https://arxiv.org/pdf/2308.07092>
- I-JEPA paper PDF: <https://arxiv.org/pdf/2301.08243>
- Local Sometria code:
  - `src/sometria/masking.py`
  - `src/sometria/models/masked.py`
  - `src/sometria/models/jepa.py`
  - `config/experiment_jepa.yaml`

## Findings

1. In upstream MAMP/MAE, `mask_ratio` means the removed/masked/target-token fraction, not the visible/context fraction.

   In upstream `model_mamp/transformer.py`, both `motion_aware_random_masking` and `random_masking` compute `len_keep = int(L * (1 - mask_ratio))`, keep the first `len_keep` shuffled tokens, build a mask where `0` is keep and `1` is remove, and compute loss as `(loss * mask).sum() / mask.sum()`. The `forward` default is `mask_ratio=0.80`, while the published run scripts use `mask90` configs for MAE and MAMP. See upstream source at <https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py> and scripts at <https://github.com/maoyunyao/MAMP/blob/main/script_pretrain_mae.sh> and <https://github.com/maoyunyao/MAMP/blob/main/script_pretrain_mamp.sh>.

2. Upstream MAMP's motion-aware masking chooses high-motion tokens as masked targets.

   The code constructs `x_orig_motion` from temporal differences, averages over coordinate channels, normalizes by max and `tau`, applies `softmax`, adds Gumbel noise to `log(prob)`, sorts ascending, and treats the large/noisy-probability tail as removed tokens. The MAMP paper says motion intensity is converted into a probability distribution that indicates the probability each embedding feature is masked, and that high-motion joints are masked with higher probability. See MAMP paper Section 3.4, especially lines describing Eq. 5-6 in <https://arxiv.org/pdf/2308.07092>.

3. Sometria's `motion_aware_mask` has the same polarity.

   Local `src/sometria/masking.py` computes `len_keep = round(L * (1.0 - mask_ratio))`, sorts ascending, returns `context=order[:, :len_keep]` and `targets=order[:, len_keep:]`. For `tau > 0`, higher score tokens get higher log probability plus Gumbel noise, so they preferentially land in `targets`. This matches MAMP's "high motion is masked" semantics, aside from implementation details: Sometria scores stored velocity-like channels, while upstream MAMP scores raw temporal differences over coordinates for masking.

4. A lower JEPA target fraction is not an obvious bug.

   MAE/MAMP and JEPA have different budgets. MAE/MAMP reconstruct input or motion values from a small visible context, so high mask ratios are the established recipe. I-JEPA instead uses a large context region and smaller target blocks: the paper says default target blocks have scale about 0.15-0.2 each, with one context block scale about 0.85-1.0 and overlapping target regions removed from context. See I-JEPA paper "Masking" in Appendix A and the main-method description in <https://arxiv.org/pdf/2301.08243>. So Sometria JEPA using `mask_ratio: 0.25` as target fraction is directionally consistent with JEPA-style "predict smaller held-out regions from ample context" rather than MAE-style "hide almost everything."

5. The likely issue is conceptual, not polarity: MAMP-style motion-aware target sampling may be a domain prior that changes what JEPA learns.

   I-JEPA emphasizes target blocks sampled at the output of the target encoder, with targets that are semantic and context that is informative yet sparse. It is not trying to preferentially choose only the most locally dynamic patches. Sometria JEPA currently reuses MAMP's sampler, so the student receives low-motion context and predicts high-motion teacher embeddings. That is plausible for action/motion data, but it is not a direct I-JEPA masking analogue. It biases the JEPA loss toward "explain dynamic moments from calmer context." If high-motion tokens are the most label-relevant regions, this can help. If the context becomes too quiet or spatially fragmented, it can make the target prediction underconditioned and may encourage shortcut/collapse behavior.

## Answer

There is no sign that Sometria inverted `mask_ratio`: in all three local objectives, `mask_ratio` is the held-out target fraction.

Using about 0.8-0.9 for MAE/MAMP and about 0.25 for JEPA is reasonable because MAE/MAMP and JEPA use different pretext losses. A high reconstruction mask is not automatically transferable to JEPA.

The thing worth testing is whether JEPA should use MAMP's motion-aware target sampler at all, or whether it should use a JEPA-specific sampler that keeps target count lower but makes targets more block-like/spread across time and DOFs. I would treat current motion-aware JEPA as a hypothesis, not a correctness bug. Good ablations would be:

- JEPA uniform random targets at the same target fraction.
- JEPA motion-aware targets at several target fractions, e.g. 0.15, 0.25, 0.40.
- JEPA block or tube targets across time/DOF, closer to I-JEPA's contiguous target blocks.
- Track `embed_std`, probe accuracy, and target-motion distribution for each run.

## Bottom line

The masking polarity is correct. MAE/MAMP high ratio plus JEPA lower target ratio makes sense. The open design question is whether "highest-motion tokens are the JEPA targets" is the right inductive bias; it is MAMP-consistent, but not I-JEPA-canonical.
