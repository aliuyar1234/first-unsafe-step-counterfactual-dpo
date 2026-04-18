# Domain map protocol

The split is stratified by **scenario stem domain**, so the domain map is a frozen input and must not be guessed.

## Procedure
1. Clone `external/ODCV-Bench` first.
2. Enumerate the actual stem directories from `external/ODCV-Bench/mandated_scenarios`.
3. Read the benchmark paper and repo metadata to recover the benchmark's real domain vocabulary.
4. Fill `configs/domain_map.yaml` with exactly one domain label per stem.
5. Commit/freeze the file before building the split.

## Rules
- Do not use `configs/domain_map.example.yaml` as an active config.
- Do not invent stem names before reading the benchmark from disk.
- Do not reassign a stem to a different domain after results exist.
- Keep mandated and incentivized variants together by stem.

## Acceptance
- Every stem in the benchmark appears exactly once.
- No extra stems exist.
- `src.splits.build_split` refuses to run if `stem_to_domain` is empty or incomplete.
