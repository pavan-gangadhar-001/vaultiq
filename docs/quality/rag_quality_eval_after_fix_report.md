# Local Doc QA RAG Evaluation Report

- Test cases: 150
- Corpus path on emulator: `/data/user/0/com.pavganga.localdocqa/app_flutter/local_doc_qa_eval_corpus.md`
- Total runtime: 68.9s
- Mean total score: 0.958
- Mean answer score: 0.947
- Retrieval hit rate: 1.000
- Citation rate: 0.927

## Chunking

- Chunks: 38
- Min chars: 304
- Max chars: 901
- Average chars: 834.0
- Complete record coverage: 100.0%

## Scores By Category

| Category | Cases | Total | Answer | Retrieval | Citation |
|---|---:|---:|---:|---:|---:|
| single-owner | 40 | 1.000 | 1.000 | 1.000 | 1.000 |
| single-deadline | 40 | 0.922 | 0.900 | 1.000 | 0.875 |
| numeric-budget | 30 | 0.925 | 0.900 | 1.000 | 0.900 |
| multi-field | 25 | 0.962 | 0.960 | 1.000 | 0.880 |
| numeric-retention | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| unanswerable | 5 | 1.000 | 1.000 | 1.000 | 1.000 |

## Per-Case Results

| # | ID | Category | Score | Retrieval | Answer | Citation | Latency ms | Expected | Output |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | OWNER-CASE-001 | single-owner | 1.00 | 1 | 1.00 | 1 | 503 | Blue Harbor | Owner team: Blue Harbor. [local_doc_qa_eval_corpus.md] |
| 2 | OWNER-CASE-002 | single-owner | 1.00 | 1 | 1.00 | 1 | 276 | Quartz Finance | Owner team: Quartz Finance. [local_doc_qa_eval_corpus.md] |
| 3 | OWNER-CASE-003 | single-owner | 1.00 | 1 | 1.00 | 1 | 313 | Cedar Security | Owner team: Cedar Security. [local_doc_qa_eval_corpus.md] |
| 4 | OWNER-CASE-004 | single-owner | 1.00 | 1 | 1.00 | 1 | 257 | Orbit Support | Owner team: Orbit Support. [local_doc_qa_eval_corpus.md] |
| 5 | OWNER-CASE-005 | single-owner | 1.00 | 1 | 1.00 | 1 | 227 | Lumen Data | Owner team: Lumen Data. [local_doc_qa_eval_corpus.md] |
| 6 | OWNER-CASE-006 | single-owner | 1.00 | 1 | 1.00 | 1 | 198 | Aster Compliance | Owner team: Aster Compliance. [local_doc_qa_eval_corpus.md] |
| 7 | OWNER-CASE-007 | single-owner | 1.00 | 1 | 1.00 | 1 | 203 | Northstar Ops | Owner team: Northstar Ops. [local_doc_qa_eval_corpus.md] |
| 8 | OWNER-CASE-008 | single-owner | 1.00 | 1 | 1.00 | 1 | 196 | Neon Platform | Owner team: Neon Platform. [local_doc_qa_eval_corpus.md] |
| 9 | OWNER-CASE-009 | single-owner | 1.00 | 1 | 1.00 | 1 | 211 | Blue Harbor | Owner team: Blue Harbor. [local_doc_qa_eval_corpus.md] |
| 10 | OWNER-CASE-010 | single-owner | 1.00 | 1 | 1.00 | 1 | 177 | Quartz Finance | Owner team: Quartz Finance. [local_doc_qa_eval_corpus.md] |
| 11 | OWNER-CASE-011 | single-owner | 1.00 | 1 | 1.00 | 1 | 242 | Cedar Security | Owner team: Cedar Security. [local_doc_qa_eval_corpus.md] |
| 12 | OWNER-CASE-012 | single-owner | 1.00 | 1 | 1.00 | 1 | 230 | Orbit Support | Owner team: Orbit Support. [local_doc_qa_eval_corpus.md] |
| 13 | OWNER-CASE-013 | single-owner | 1.00 | 1 | 1.00 | 1 | 197 | Lumen Data | Owner team: Lumen Data. [local_doc_qa_eval_corpus.md] |
| 14 | OWNER-CASE-014 | single-owner | 1.00 | 1 | 1.00 | 1 | 209 | Aster Compliance | Owner team: Aster Compliance. [local_doc_qa_eval_corpus.md] |
| 15 | OWNER-CASE-015 | single-owner | 1.00 | 1 | 1.00 | 1 | 192 | Northstar Ops | Owner team: Northstar Ops. [local_doc_qa_eval_corpus.md] |
| 16 | OWNER-CASE-016 | single-owner | 1.00 | 1 | 1.00 | 1 | 192 | Neon Platform | Owner team: Neon Platform. [local_doc_qa_eval_corpus.md] |
| 17 | OWNER-CASE-017 | single-owner | 1.00 | 1 | 1.00 | 1 | 219 | Blue Harbor | Owner team: Blue Harbor. [local_doc_qa_eval_corpus.md] |
| 18 | OWNER-CASE-018 | single-owner | 1.00 | 1 | 1.00 | 1 | 387 | Quartz Finance | Owner team: Quartz Finance. [local_doc_qa_eval_corpus.md] |
| 19 | OWNER-CASE-019 | single-owner | 1.00 | 1 | 1.00 | 1 | 478 | Cedar Security | Owner team: Cedar Security. [local_doc_qa_eval_corpus.md] |
| 20 | OWNER-CASE-020 | single-owner | 1.00 | 1 | 1.00 | 1 | 264 | Orbit Support | Owner team: Orbit Support. [local_doc_qa_eval_corpus.md] |
| 21 | OWNER-CASE-021 | single-owner | 1.00 | 1 | 1.00 | 1 | 252 | Lumen Data | Owner team: Lumen Data. [local_doc_qa_eval_corpus.md] |
| 22 | OWNER-CASE-022 | single-owner | 1.00 | 1 | 1.00 | 1 | 201 | Aster Compliance | Owner team: Aster Compliance. [local_doc_qa_eval_corpus.md] |
| 23 | OWNER-CASE-023 | single-owner | 1.00 | 1 | 1.00 | 1 | 193 | Northstar Ops | Owner team: Northstar Ops. [local_doc_qa_eval_corpus.md] |
| 24 | OWNER-CASE-024 | single-owner | 1.00 | 1 | 1.00 | 1 | 174 | Neon Platform | Owner team: Neon Platform. [local_doc_qa_eval_corpus.md] |
| 25 | OWNER-CASE-025 | single-owner | 1.00 | 1 | 1.00 | 1 | 184 | Blue Harbor | Owner team: Blue Harbor. [local_doc_qa_eval_corpus.md] |
| 26 | OWNER-CASE-026 | single-owner | 1.00 | 1 | 1.00 | 1 | 231 | Quartz Finance | Owner team: Quartz Finance. [local_doc_qa_eval_corpus.md] |
| 27 | OWNER-CASE-027 | single-owner | 1.00 | 1 | 1.00 | 1 | 321 | Cedar Security | Owner team: Cedar Security. [local_doc_qa_eval_corpus.md] |
| 28 | OWNER-CASE-028 | single-owner | 1.00 | 1 | 1.00 | 1 | 298 | Orbit Support | Owner team: Orbit Support. [local_doc_qa_eval_corpus.md] |
| 29 | OWNER-CASE-029 | single-owner | 1.00 | 1 | 1.00 | 1 | 201 | Lumen Data | Owner team: Lumen Data. [local_doc_qa_eval_corpus.md] |
| 30 | OWNER-CASE-030 | single-owner | 1.00 | 1 | 1.00 | 1 | 174 | Aster Compliance | Owner team: Aster Compliance. [local_doc_qa_eval_corpus.md] |
| 31 | OWNER-CASE-031 | single-owner | 1.00 | 1 | 1.00 | 1 | 189 | Northstar Ops | Owner team: Northstar Ops. [local_doc_qa_eval_corpus.md] |
| 32 | OWNER-CASE-032 | single-owner | 1.00 | 1 | 1.00 | 1 | 175 | Neon Platform | Owner team: Neon Platform. [local_doc_qa_eval_corpus.md] |
| 33 | OWNER-CASE-033 | single-owner | 1.00 | 1 | 1.00 | 1 | 183 | Blue Harbor | Owner team: Blue Harbor. [local_doc_qa_eval_corpus.md] |
| 34 | OWNER-CASE-034 | single-owner | 1.00 | 1 | 1.00 | 1 | 164 | Quartz Finance | Owner team: Quartz Finance. [local_doc_qa_eval_corpus.md] |
| 35 | OWNER-CASE-035 | single-owner | 1.00 | 1 | 1.00 | 1 | 164 | Cedar Security | Owner team: Cedar Security. [local_doc_qa_eval_corpus.md] |
| 36 | OWNER-CASE-036 | single-owner | 1.00 | 1 | 1.00 | 1 | 210 | Orbit Support | Owner team: Orbit Support. [local_doc_qa_eval_corpus.md] |
| 37 | OWNER-CASE-037 | single-owner | 1.00 | 1 | 1.00 | 1 | 196 | Lumen Data | Owner team: Lumen Data. [local_doc_qa_eval_corpus.md] |
| 38 | OWNER-CASE-038 | single-owner | 1.00 | 1 | 1.00 | 1 | 184 | Aster Compliance | Owner team: Aster Compliance. [local_doc_qa_eval_corpus.md] |
| 39 | OWNER-CASE-039 | single-owner | 1.00 | 1 | 1.00 | 1 | 176 | Northstar Ops | Owner team: Northstar Ops. [local_doc_qa_eval_corpus.md] |
| 40 | OWNER-CASE-040 | single-owner | 1.00 | 1 | 1.00 | 1 | 227 | Neon Platform | Owner team: Neon Platform. [local_doc_qa_eval_corpus.md] |
| 41 | DEADLINE-CASE-011 | single-deadline | 1.00 | 1 | 1.00 | 1 | 234 | 2026-03-07 | Deadline: 2026-03-07. [local_doc_qa_eval_corpus.md] |
| 42 | DEADLINE-CASE-012 | single-deadline | 0.25 | 1 | 0.00 | 0 | 4594 | 2026-04-10 | The exact deadline for CASE-012 is 2026-02-04. |
| 43 | DEADLINE-CASE-013 | single-deadline | 1.00 | 1 | 1.00 | 1 | 229 | 2026-05-13 | Deadline: 2026-05-13. [local_doc_qa_eval_corpus.md] |
| 44 | DEADLINE-CASE-014 | single-deadline | 1.00 | 1 | 1.00 | 1 | 215 | 2026-06-16 | Deadline: 2026-06-16. [local_doc_qa_eval_corpus.md] |
| 45 | DEADLINE-CASE-015 | single-deadline | 1.00 | 1 | 1.00 | 1 | 204 | 2026-07-19 | Deadline: 2026-07-19. [local_doc_qa_eval_corpus.md] |
| 46 | DEADLINE-CASE-016 | single-deadline | 1.00 | 1 | 1.00 | 1 | 239 | 2026-08-22 | Deadline: 2026-08-22. [local_doc_qa_eval_corpus.md] |
| 47 | DEADLINE-CASE-017 | single-deadline | 1.00 | 1 | 1.00 | 1 | 202 | 2026-09-25 | Deadline: 2026-09-25. [local_doc_qa_eval_corpus.md] |
| 48 | DEADLINE-CASE-018 | single-deadline | 1.00 | 1 | 1.00 | 1 | 175 | 2026-01-01 | Deadline: 2026-01-01. [local_doc_qa_eval_corpus.md] |
| 49 | DEADLINE-CASE-019 | single-deadline | 1.00 | 1 | 1.00 | 1 | 169 | 2026-02-04 | Deadline: 2026-02-04. [local_doc_qa_eval_corpus.md] |
| 50 | DEADLINE-CASE-020 | single-deadline | 0.25 | 1 | 0.00 | 0 | 3612 | 2026-03-07 | The exact deadline for CASE-020 is 2026-02-04. |
| 51 | DEADLINE-CASE-021 | single-deadline | 1.00 | 1 | 1.00 | 1 | 294 | 2026-04-10 | Deadline: 2026-04-10. [local_doc_qa_eval_corpus.md] |
| 52 | DEADLINE-CASE-022 | single-deadline | 1.00 | 1 | 1.00 | 1 | 228 | 2026-05-13 | Deadline: 2026-05-13. [local_doc_qa_eval_corpus.md] |
| 53 | DEADLINE-CASE-023 | single-deadline | 1.00 | 1 | 1.00 | 1 | 201 | 2026-06-16 | Deadline: 2026-06-16. [local_doc_qa_eval_corpus.md] |
| 54 | DEADLINE-CASE-024 | single-deadline | 1.00 | 1 | 1.00 | 1 | 206 | 2026-07-19 | Deadline: 2026-07-19. [local_doc_qa_eval_corpus.md] |
| 55 | DEADLINE-CASE-025 | single-deadline | 0.25 | 1 | 0.00 | 0 | 2441 | 2026-08-22 | The exact deadline for CASE-025 is 2026-02-04. |
| 56 | DEADLINE-CASE-026 | single-deadline | 1.00 | 1 | 1.00 | 1 | 298 | 2026-09-25 | Deadline: 2026-09-25. [local_doc_qa_eval_corpus.md] |
| 57 | DEADLINE-CASE-027 | single-deadline | 1.00 | 1 | 1.00 | 1 | 402 | 2026-01-01 | Deadline: 2026-01-01. [local_doc_qa_eval_corpus.md] |
| 58 | DEADLINE-CASE-028 | single-deadline | 1.00 | 1 | 1.00 | 1 | 537 | 2026-02-04 | Deadline: 2026-02-04. [local_doc_qa_eval_corpus.md] |
| 59 | DEADLINE-CASE-029 | single-deadline | 1.00 | 1 | 1.00 | 1 | 345 | 2026-03-07 | Deadline: 2026-03-07. [local_doc_qa_eval_corpus.md] |
| 60 | DEADLINE-CASE-030 | single-deadline | 1.00 | 1 | 1.00 | 1 | 247 | 2026-04-10 | Deadline: 2026-04-10. [local_doc_qa_eval_corpus.md] |
| 61 | DEADLINE-CASE-031 | single-deadline | 1.00 | 1 | 1.00 | 1 | 289 | 2026-05-13 | Deadline: 2026-05-13. [local_doc_qa_eval_corpus.md] |
| 62 | DEADLINE-CASE-032 | single-deadline | 1.00 | 1 | 1.00 | 1 | 309 | 2026-06-16 | Deadline: 2026-06-16. [local_doc_qa_eval_corpus.md] |
| 63 | DEADLINE-CASE-033 | single-deadline | 1.00 | 1 | 1.00 | 1 | 248 | 2026-07-19 | Deadline: 2026-07-19. [local_doc_qa_eval_corpus.md] |
| 64 | DEADLINE-CASE-034 | single-deadline | 1.00 | 1 | 1.00 | 1 | 215 | 2026-08-22 | Deadline: 2026-08-22. [local_doc_qa_eval_corpus.md] |
| 65 | DEADLINE-CASE-035 | single-deadline | 1.00 | 1 | 1.00 | 1 | 219 | 2026-09-25 | Deadline: 2026-09-25. [local_doc_qa_eval_corpus.md] |
| 66 | DEADLINE-CASE-036 | single-deadline | 1.00 | 1 | 1.00 | 1 | 209 | 2026-01-01 | Deadline: 2026-01-01. [local_doc_qa_eval_corpus.md] |
| 67 | DEADLINE-CASE-037 | single-deadline | 1.00 | 1 | 1.00 | 1 | 209 | 2026-02-04 | Deadline: 2026-02-04. [local_doc_qa_eval_corpus.md] |
| 68 | DEADLINE-CASE-038 | single-deadline | 1.00 | 1 | 1.00 | 1 | 194 | 2026-03-07 | Deadline: 2026-03-07. [local_doc_qa_eval_corpus.md] |
| 69 | DEADLINE-CASE-039 | single-deadline | 1.00 | 1 | 1.00 | 1 | 196 | 2026-04-10 | Deadline: 2026-04-10. [local_doc_qa_eval_corpus.md] |
| 70 | DEADLINE-CASE-040 | single-deadline | 1.00 | 1 | 1.00 | 1 | 349 | 2026-05-13 | Deadline: 2026-05-13. [local_doc_qa_eval_corpus.md] |
| 71 | DEADLINE-CASE-041 | single-deadline | 1.00 | 1 | 1.00 | 1 | 327 | 2026-06-16 | Deadline: 2026-06-16. [local_doc_qa_eval_corpus.md] |
| 72 | DEADLINE-CASE-042 | single-deadline | 1.00 | 1 | 1.00 | 1 | 208 | 2026-07-19 | Deadline: 2026-07-19. [local_doc_qa_eval_corpus.md] |
| 73 | DEADLINE-CASE-043 | single-deadline | 0.25 | 1 | 0.00 | 0 | 3391 | 2026-08-22 | The exact deadline for CASE-043 is 2026-02-04. |
| 74 | DEADLINE-CASE-044 | single-deadline | 1.00 | 1 | 1.00 | 1 | 369 | 2026-09-25 | Deadline: 2026-09-25. [local_doc_qa_eval_corpus.md] |
| 75 | DEADLINE-CASE-045 | single-deadline | 1.00 | 1 | 1.00 | 1 | 278 | 2026-01-01 | Deadline: 2026-01-01. [local_doc_qa_eval_corpus.md] |
| 76 | DEADLINE-CASE-046 | single-deadline | 0.90 | 1 | 1.00 | 0 | 2622 | 2026-02-04 | The exact deadline for CASE-046 is 2026-02-04. |
| 77 | DEADLINE-CASE-047 | single-deadline | 1.00 | 1 | 1.00 | 1 | 366 | 2026-03-07 | Deadline: 2026-03-07. [local_doc_qa_eval_corpus.md] |
| 78 | DEADLINE-CASE-048 | single-deadline | 1.00 | 1 | 1.00 | 1 | 269 | 2026-04-10 | Deadline: 2026-04-10. [local_doc_qa_eval_corpus.md] |
| 79 | DEADLINE-CASE-049 | single-deadline | 1.00 | 1 | 1.00 | 1 | 235 | 2026-05-13 | Deadline: 2026-05-13. [local_doc_qa_eval_corpus.md] |
| 80 | DEADLINE-CASE-050 | single-deadline | 1.00 | 1 | 1.00 | 1 | 213 | 2026-06-16 | Deadline: 2026-06-16. [local_doc_qa_eval_corpus.md] |
| 81 | BUDGET-CASE-021 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 225 | 2187 | Budget cap: $2187. [local_doc_qa_eval_corpus.md] |
| 82 | BUDGET-CASE-022 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 206 | 2234 | Budget cap: $2234. [local_doc_qa_eval_corpus.md] |
| 83 | BUDGET-CASE-023 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 188 | 2281 | Budget cap: $2281. [local_doc_qa_eval_corpus.md] |
| 84 | BUDGET-CASE-024 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 213 | 2328 | Budget cap: $2328. [local_doc_qa_eval_corpus.md] |
| 85 | BUDGET-CASE-025 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5024 | 2375 | $1247 |
| 86 | BUDGET-CASE-026 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 359 | 2422 | Budget cap: $2422. [local_doc_qa_eval_corpus.md] |
| 87 | BUDGET-CASE-027 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 308 | 2469 | Budget cap: $2469. [local_doc_qa_eval_corpus.md] |
| 88 | BUDGET-CASE-028 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 318 | 2516 | Budget cap: $2516. [local_doc_qa_eval_corpus.md] |
| 89 | BUDGET-CASE-029 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 313 | 2563 | Budget cap: $2563. [local_doc_qa_eval_corpus.md] |
| 90 | BUDGET-CASE-030 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 227 | 2610 | Budget cap: $2610. [local_doc_qa_eval_corpus.md] |
| 91 | BUDGET-CASE-031 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 236 | 2657 | Budget cap: $2657. [local_doc_qa_eval_corpus.md] |
| 92 | BUDGET-CASE-032 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 253 | 2704 | Budget cap: $2704. [local_doc_qa_eval_corpus.md] |
| 93 | BUDGET-CASE-033 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 234 | 2751 | Budget cap: $2751. [local_doc_qa_eval_corpus.md] |
| 94 | BUDGET-CASE-034 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 217 | 2798 | Budget cap: $2798. [local_doc_qa_eval_corpus.md] |
| 95 | BUDGET-CASE-035 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 318 | 2845 | Budget cap: $2845. [local_doc_qa_eval_corpus.md] |
| 96 | BUDGET-CASE-036 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 427 | 2892 | Budget cap: $2892. [local_doc_qa_eval_corpus.md] |
| 97 | BUDGET-CASE-037 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 360 | 2939 | Budget cap: $2939. [local_doc_qa_eval_corpus.md] |
| 98 | BUDGET-CASE-038 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 234 | 2986 | Budget cap: $2986. [local_doc_qa_eval_corpus.md] |
| 99 | BUDGET-CASE-039 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 210 | 3033 | Budget cap: $3033. [local_doc_qa_eval_corpus.md] |
| 100 | BUDGET-CASE-040 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 231 | 3080 | Budget cap: $3080. [local_doc_qa_eval_corpus.md] |
| 101 | BUDGET-CASE-041 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 198 | 3127 | Budget cap: $3127. [local_doc_qa_eval_corpus.md] |
| 102 | BUDGET-CASE-042 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 189 | 3174 | Budget cap: $3174. [local_doc_qa_eval_corpus.md] |
| 103 | BUDGET-CASE-043 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 2112 | 3221 | $1247 |
| 104 | BUDGET-CASE-044 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 289 | 3268 | Budget cap: $3268. [local_doc_qa_eval_corpus.md] |
| 105 | BUDGET-CASE-045 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 245 | 3315 | Budget cap: $3315. [local_doc_qa_eval_corpus.md] |
| 106 | BUDGET-CASE-046 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 2495 | 3362 | $1247 |
| 107 | BUDGET-CASE-047 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 546 | 3409 | Budget cap: $3409. [local_doc_qa_eval_corpus.md] |
| 108 | BUDGET-CASE-048 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 350 | 3456 | Budget cap: $3456. [local_doc_qa_eval_corpus.md] |
| 109 | BUDGET-CASE-049 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 265 | 3503 | Budget cap: $3503. [local_doc_qa_eval_corpus.md] |
| 110 | BUDGET-CASE-050 | numeric-budget | 1.00 | 1 | 1.00 | 1 | 258 | 3550 | Budget cap: $3550. [local_doc_qa_eval_corpus.md] |
| 111 | MULTI-CASE-001 | multi-field | 1.00 | 1 | 1.00 | 1 | 228 | finance director approval; Seattle | Approval path: finance director approval. Primary region: Seattle. [local_doc_qa_eval_corpus.md] |
| 112 | MULTI-CASE-003 | multi-field | 1.00 | 1 | 1.00 | 1 | 230 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 113 | MULTI-CASE-005 | multi-field | 1.00 | 1 | 1.00 | 1 | 193 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 114 | MULTI-CASE-007 | multi-field | 0.90 | 1 | 1.00 | 0 | 2055 | finance director approval; Seattle | The approval path for CASE-007 is finance director approval, and the primary region is Seattle. |
| 115 | MULTI-CASE-009 | multi-field | 1.00 | 1 | 1.00 | 1 | 246 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 116 | MULTI-CASE-011 | multi-field | 1.00 | 1 | 1.00 | 1 | 220 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 117 | MULTI-CASE-013 | multi-field | 1.00 | 1 | 1.00 | 1 | 190 | finance director approval; Seattle | Approval path: finance director approval. Primary region: Seattle. [local_doc_qa_eval_corpus.md] |
| 118 | MULTI-CASE-015 | multi-field | 1.00 | 1 | 1.00 | 1 | 206 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 119 | MULTI-CASE-017 | multi-field | 1.00 | 1 | 1.00 | 1 | 186 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 120 | MULTI-CASE-019 | multi-field | 1.00 | 1 | 1.00 | 1 | 177 | finance director approval; Seattle | Approval path: finance director approval. Primary region: Seattle. [local_doc_qa_eval_corpus.md] |
| 121 | MULTI-CASE-021 | multi-field | 1.00 | 1 | 1.00 | 1 | 272 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 122 | MULTI-CASE-023 | multi-field | 1.00 | 1 | 1.00 | 1 | 516 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 123 | MULTI-CASE-025 | multi-field | 0.25 | 1 | 0.00 | 0 | 2514 | finance director approval; Seattle | Primary region: Dublin. Approval path: legal review. |
| 124 | MULTI-CASE-027 | multi-field | 1.00 | 1 | 1.00 | 1 | 317 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 125 | MULTI-CASE-029 | multi-field | 1.00 | 1 | 1.00 | 1 | 284 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 126 | MULTI-CASE-031 | multi-field | 1.00 | 1 | 1.00 | 1 | 225 | finance director approval; Seattle | Approval path: finance director approval. Primary region: Seattle. [local_doc_qa_eval_corpus.md] |
| 127 | MULTI-CASE-033 | multi-field | 1.00 | 1 | 1.00 | 1 | 207 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 128 | MULTI-CASE-035 | multi-field | 1.00 | 1 | 1.00 | 1 | 208 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 129 | MULTI-CASE-037 | multi-field | 1.00 | 1 | 1.00 | 1 | 201 | finance director approval; Seattle | Approval path: finance director approval. Primary region: Seattle. [local_doc_qa_eval_corpus.md] |
| 130 | MULTI-CASE-039 | multi-field | 1.00 | 1 | 1.00 | 1 | 348 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 131 | MULTI-CASE-041 | multi-field | 1.00 | 1 | 1.00 | 1 | 318 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 132 | MULTI-CASE-043 | multi-field | 0.90 | 1 | 1.00 | 0 | 3319 | finance director approval; Seattle | The approval path for CASE-043 is finance director approval, and the primary region is Seattle. |
| 133 | MULTI-CASE-045 | multi-field | 1.00 | 1 | 1.00 | 1 | 253 | legal review; Singapore | Approval path: legal review. Primary region: Singapore. [local_doc_qa_eval_corpus.md] |
| 134 | MULTI-CASE-047 | multi-field | 1.00 | 1 | 1.00 | 1 | 252 | incident commander approval; Berlin | Approval path: incident commander approval. Primary region: Berlin. [local_doc_qa_eval_corpus.md] |
| 135 | MULTI-CASE-049 | multi-field | 1.00 | 1 | 1.00 | 1 | 220 | finance director approval; Seattle | Approval path: finance director approval. Primary region: Seattle. [local_doc_qa_eval_corpus.md] |
| 136 | RETENTION-CASE-004 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 196 | 18; days | Retention window: 18 days. [local_doc_qa_eval_corpus.md] |
| 137 | RETENTION-CASE-009 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 182 | 23; days | Retention window: 23 days. [local_doc_qa_eval_corpus.md] |
| 138 | RETENTION-CASE-014 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 175 | 28; days | Retention window: 28 days. [local_doc_qa_eval_corpus.md] |
| 139 | RETENTION-CASE-019 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 174 | 33; days | Retention window: 33 days. [local_doc_qa_eval_corpus.md] |
| 140 | RETENTION-CASE-024 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 180 | 17; days | Retention window: 17 days. [local_doc_qa_eval_corpus.md] |
| 141 | RETENTION-CASE-029 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 397 | 22; days | Retention window: 22 days. [local_doc_qa_eval_corpus.md] |
| 142 | RETENTION-CASE-034 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 320 | 27; days | Retention window: 27 days. [local_doc_qa_eval_corpus.md] |
| 143 | RETENTION-CASE-039 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 231 | 32; days | Retention window: 32 days. [local_doc_qa_eval_corpus.md] |
| 144 | RETENTION-CASE-044 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 205 | 16; days | Retention window: 16 days. [local_doc_qa_eval_corpus.md] |
| 145 | RETENTION-CASE-049 | numeric-retention | 1.00 | 1 | 1.00 | 1 | 219 | 21; days | Retention window: 21 days. [local_doc_qa_eval_corpus.md] |
| 146 | MISSING-CASE-090 | unanswerable | 1.00 | 1 | 1.00 | 1 | 188 | missing | CASE-090 is missing from the local context. [local_doc_qa_eval_corpus.md] |
| 147 | MISSING-CASE-091 | unanswerable | 1.00 | 1 | 1.00 | 1 | 200 | missing | CASE-091 is missing from the local context. [local_doc_qa_eval_corpus.md] |
| 148 | MISSING-CASE-092 | unanswerable | 1.00 | 1 | 1.00 | 1 | 182 | missing | CASE-092 is missing from the local context. [local_doc_qa_eval_corpus.md] |
| 149 | MISSING-CASE-093 | unanswerable | 1.00 | 1 | 1.00 | 1 | 193 | missing | CASE-093 is missing from the local context. [local_doc_qa_eval_corpus.md] |
| 150 | MISSING-CASE-094 | unanswerable | 1.00 | 1 | 1.00 | 1 | 222 | missing | CASE-094 is missing from the local context. [local_doc_qa_eval_corpus.md] |
