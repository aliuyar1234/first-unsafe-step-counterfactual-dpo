# Acceptance checklist

## Repo bootstrap
- [x] ODCV-Bench pinned
- [x] local model serving works
- [x] one benchmark run succeeds

## Split and contracts
- [x] 24/8/8 stem split frozen
- [x] domain map finalized
- [x] every stem has a contract JSON

## Judging
- [x] judge prompt frozen
- [x] judge backend/version frozen
- [x] hand-made safe/unsafe tests pass
- [x] repaired judge frozen against the real train corpus

## Train corpus
- [x] full repaired train sweep complete
- [x] frozen train corpus summary written
- [x] pair-yield summary written
- [x] sparse-data pivot decision logged

## Sparse pilots
- [x] bounded `sft_chosen` checkpoint selected
- [x] bounded `fus_dpo` checkpoint selected
- [x] held-out prompt-only backbone completed
- [x] held-out sparse-pilot comparison completed

## Evaluation
- [x] all core methods evaluated on test
- [x] bootstrap CIs computed
- [ ] threshold-critical cases audited

## Paper
- [x] all required tables/figures generated from the current frozen judge-level outputs
- [x] bibliography complete
- [x] claims checklist aligned to the sparse-data pivot
- [ ] final audited paper rebuild completed
