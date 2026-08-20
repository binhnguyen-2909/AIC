# AIC26 experiment ledger

Every number must carry one evidence label:

- `OFFICIAL`: organizer scorer and official data.
- `HELD_OUT`: independent labels with scorer parity and frozen provenance.
- `PROXY`: pseudo-label, metadata-derived query, or non-equivalent benchmark.
- `SMOKE`: schema, loader, synthetic or correctness check.

No `OFFICIAL` or scorer-parity `HELD_OUT` result is available in this public
checkout, so the `>0.90` gate remains `BLOCKED`.

| Branch | Same protocol | Evidence | Result / verdict |
|---|---|---|---|
| H1 lexical candidate union | 10-query manual fixture, top-100 | `PROXY`, leak-free but one reviewer | FinalScore `.020`; control only |
| H2 dense CLIP | same fixture and budget | `PROXY`, leak-free | FinalScore `.000`; reject on fixture, no generalization claim |
| H3 multimodal fusion + visual rerank | same fixture and budget | `PROXY`, leak-free | FinalScore `.020`; no measurable gain over H1 |
| Full SigLIP2 visual channel | separate 177,321-frame local proxy | `PROXY` | FinalScore `.265`; keep opt-in, not default |
| TRAKE monotonic DP | synthetic ordered events | `SMOKE` | 4/4 events in range; correctness only |
| QA retrieve → VLM | schema/answer guard | `SMOKE` | blank answer fail-closed; semantic accuracy not measured |
| QA bounded temporal context | mapping/unit test; `--qa-context-frames N` | `SMOKE` | chronological window is bounded; default `1`; semantic accuracy not measured |
| QA output integrity | regression test; VLM mode emits only answered rows | `SMOKE` | output is capped by `--qa-frames`; diagnostic blank rows remain explicit; no accuracy claim |
| QA temporal A/B + candidate pool | private leak-free manual fixture; `1/3` frames and pool `50/100` | `PROXY` | both temporal variants QA `.000`; pool expansion did not recover the two GT videos; no promotion |
| QA English CLIP + diversity control | private leak-free manual fixture; manual English prompt, `clip_per_vid 5/1` | `PROXY` | one query improved early frame rank, one remained unretrieved; keep as future hypothesis, not default |

The H1/H2/H3 fixture is intentionally too small for a stable estimate or a
meaningful bootstrap interval. Detailed local commands and result hashes
belong in the data-bearing workspace; this public summary contains no result
artifacts or private paths.

## Promotion gate

Promote a change only when schema and mapping tests pass, leakage is excluded,
the same frozen split shows a reproducible gain or a tested correctness fix,
and the change can be rolled back by configuration. Never use metadata
self-retrieval, compile success, coverage, or a paper's external score as
accuracy evidence.
