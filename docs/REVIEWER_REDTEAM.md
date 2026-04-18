# Reviewer red team

This file exists so implementation stays aligned with likely reviewer attacks.

## Top objections and required evidence

1. **"This is just generic DPO."**
   - Required evidence: localized first-unsafe-step construction and SFT ablation.

2. **"You overfit to ODCV-Bench."**
   - Required evidence: stem-level held-out split, frozen test, no leakage.

3. **"Prompt baselines are weak."**
   - Required evidence: both `contract_prompt` and `refusal_prompt`.

4. **"Your gains are just overrefusal."**
   - Required evidence: `KPI-SR`, `Safe-KPI-SR`, `RefusalRate`, tradeoff figure.

5. **"AI judging is unreliable."**
   - Required evidence: threshold audit, frozen judge version, disagreement policy.

6. **"One model is not enough."**
   - Required response: narrow v1 scope; do not claim cross-model generalization.

7. **"Why not runtime enforcement?"**
   - Required response: this paper is a narrow post-training study, not a runtime-defense comparison.

8. **"The first unsafe step is ambiguous."**
   - Required evidence: localization protocol and manual review for threshold cases.

9. **"Utility is underspecified."**
   - Required evidence: precise KPI-SR / Safe-KPI-SR definitions.

10. **"The training data is synthetic and noisy."**
    - Required evidence: filters, offline audit, pair-quality stats.

11. **"This duplicates MOSAIC / ToolSafe / Thought-Aligner / AgentSpec."**
    - Required response: novelty boundary is ODCV-specific, held-out, localized next-step DPO on one 7B model.

12. **"The benchmark is too small."**
    - Required evidence: multiple seeds, per-stem reporting, careful held-out evaluation.

13. **"Your method may memorize safe templates."**
    - Required evidence: SFT ablation and no-contract ablation if time.

14. **"The paper overclaims."**
    - Required response: keep claims narrow; never say “solves”.

15. **"The result is too modest."**
    - Required response: emphasize reproducible evaluation and where localized training fails or helps.

## Claims you must not make
- solves KPI-gaming
- generalizes to all agent safety
- beats runtime defenses
- provides guarantees
- works across all models
