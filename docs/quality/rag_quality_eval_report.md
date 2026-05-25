# Local Doc QA RAG Evaluation Report

- Test cases: 150
- Corpus path on emulator: `/data/user/0/com.pavganga.localdocqa/app_flutter/local_doc_qa_eval_corpus.md`
- Total runtime: 1031.9s
- Mean total score: 0.318
- Mean answer score: 0.057
- Retrieval hit rate: 1.000
- Citation rate: 0.313

## Chunking

- Chunks: 38
- Min chars: 304
- Max chars: 901
- Average chars: 834.0
- Complete record coverage: 100.0%

## Scores By Category

| Category | Cases | Total | Answer | Retrieval | Citation |
|---|---:|---:|---:|---:|---:|
| single-owner | 40 | 0.366 | 0.025 | 1.000 | 1.000 |
| single-deadline | 40 | 0.255 | 0.000 | 1.000 | 0.050 |
| numeric-budget | 30 | 0.250 | 0.000 | 1.000 | 0.000 |
| multi-field | 25 | 0.344 | 0.120 | 1.000 | 0.160 |
| numeric-retention | 10 | 0.543 | 0.450 | 1.000 | 0.000 |
| unanswerable | 5 | 0.270 | 0.000 | 1.000 | 0.200 |

## Per-Case Results

| # | ID | Category | Score | Retrieval | Answer | Citation | Latency ms | Expected | Output |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | OWNER-CASE-001 | single-owner | 0.35 | 1 | 0.00 | 1 | 7025 | Blue Harbor | [notes.md] |
| 2 | OWNER-CASE-002 | single-owner | 0.35 | 1 | 0.00 | 1 | 36076 | Quartz Finance | [1] local_doc_qa_eval_corpus.md chunk 1 # Local Doc QA Evaluation Corpus This file is a synthetic evaluation corpus for local document question answering. Every record below is ... |
| 3 | OWNER-CASE-003 | single-owner | 0.35 | 1 | 0.00 | 1 | 32908 | Cedar Security | [1] local_doc_qa_eval_corpus.md chunk 2 # Local Doc QA Evaluation Corpus This file is a synthetic evaluation corpus for local document question answering. Every record below is ... |
| 4 | OWNER-CASE-004 | single-owner | 0.35 | 1 | 0.00 | 1 | 5988 | Orbit Support | [local_doc_qa_eval_corpus.md chunk 3] |
| 5 | OWNER-CASE-005 | single-owner | 0.35 | 1 | 0.00 | 1 | 5782 | Lumen Data | [local_doc_qa_eval_corpus.md chunk 3] |
| 6 | OWNER-CASE-006 | single-owner | 0.35 | 1 | 0.00 | 1 | 5840 | Aster Compliance | [local_doc_qa_eval_corpus.md chunk 4] |
| 7 | OWNER-CASE-007 | single-owner | 0.35 | 1 | 0.00 | 1 | 4460 | Northstar Ops | [local_doc_qa_eval_corpus.md chunk 4] |
| 8 | OWNER-CASE-008 | single-owner | 0.35 | 1 | 0.00 | 1 | 6083 | Neon Platform | [owner_team] Granite Sync 6 |
| 9 | OWNER-CASE-009 | single-owner | 0.35 | 1 | 0.00 | 1 | 5914 | Blue Harbor | [local_doc_qa_eval_corpus.md chunk 6] [notes.md] |
| 10 | OWNER-CASE-010 | single-owner | 0.35 | 1 | 0.00 | 1 | 5766 | Quartz Finance | [local_doc_qa_eval_corpus.md chunk 1] |
| 11 | OWNER-CASE-011 | single-owner | 0.35 | 1 | 0.00 | 1 | 5158 | Cedar Security | [local_doc_qa_eval_corpus.md chunk 1] |
| 12 | OWNER-CASE-012 | single-owner | 0.35 | 1 | 0.00 | 1 | 4621 | Orbit Support | [local_doc_qa_eval_corpus.md chunk 8] |
| 13 | OWNER-CASE-013 | single-owner | 0.35 | 1 | 0.00 | 1 | 4815 | Lumen Data | [owner team] Delta Archive 13 |
| 14 | OWNER-CASE-014 | single-owner | 0.35 | 1 | 0.00 | 1 | 6915 | Aster Compliance | [owner team: Orbit Support, local_doc_qa_eval_corpus.md chunk 9] |
| 15 | OWNER-CASE-015 | single-owner | 0.35 | 1 | 0.00 | 1 | 5882 | Northstar Ops | [owner team] CASE-015 |
| 16 | OWNER-CASE-016 | single-owner | 0.35 | 1 | 0.00 | 1 | 5484 | Neon Platform | [Owner team: Granite Sync 16, Local Doc QA Evaluation Corpus.md chunk 11] |
| 17 | OWNER-CASE-017 | single-owner | 0.35 | 1 | 0.00 | 1 | 15814 | Blue Harbor | [1] local_doc_qa_eval_corpus.md chunk 1 eway 15 Record ID: CASE-015. Project codename: Frost Gateway 15. Owner team: Northstar Ops. Approval path: legal review. Primary region: ... |
| 18 | OWNER-CASE-018 | single-owner | 0.35 | 1 | 0.00 | 1 | 26313 | Quartz Finance | [1] local_doc_qa_eval_corpus.md chunk 1 # Local Doc QA Evaluation Corpus This file is a synthetic evaluation corpus for local document question answering. Every record below is ... |
| 19 | OWNER-CASE-019 | single-owner | 0.35 | 1 | 0.00 | 1 | 5965 | Cedar Security | [local_doc_qa_eval_corpus.md chunk 12] |
| 20 | OWNER-CASE-020 | single-owner | 0.35 | 1 | 0.00 | 1 | 6265 | Orbit Support | [local_doc_qa_eval_corpus.md chunk 13] |
| 21 | OWNER-CASE-021 | single-owner | 1.00 | 1 | 1.00 | 1 | 17185 | Lumen Data | [1] local_doc_qa_eval_corpus.md chunk 13 al. Primary region: Bengaluru. Deadline: 2026-01-01. Budget cap: $2046. Retention window: 32 days. Severity class: high. Operational not... |
| 22 | OWNER-CASE-022 | single-owner | 0.35 | 1 | 0.00 | 1 | 12300 | Aster Compliance | [1] local_doc_qa_eval_corpus.md chunk 14 section. ## CASE-022: Atlas Relay 22 Record ID: CASE-022. Project codename: Atlas Relay 22. Owner team: Orbit Support. Approval path: se... |
| 23 | OWNER-CASE-023 | single-owner | 0.35 | 1 | 0.00 | 1 | 9692 | Northstar Ops | [local_doc_qa_eval_corpus.md chunk 14] section. ## CASE-023: Atlas Relay 23 Record ID: CASE-023. Project codename: Atlas Relay 23. Owner team: Orbit Support. Approval path: secu... |
| 24 | OWNER-CASE-024 | single-owner | 0.35 | 1 | 0.00 | 1 | 5990 | Neon Platform | [owner_team=Cobalt Bridge 22, file=local_doc_qa_eval_corpus.md chunk 15] |
| 25 | OWNER-CASE-025 | single-owner | 0.35 | 1 | 0.00 | 1 | 6484 | Blue Harbor | [owner_team] [local_doc_qa_eval_corpus.md chunk 16] |
| 26 | OWNER-CASE-026 | single-owner | 0.35 | 1 | 0.00 | 1 | 5842 | Quartz Finance | [owner team] Granite Sync 26 |
| 27 | OWNER-CASE-027 | single-owner | 0.35 | 1 | 0.00 | 1 | 9847 | Cedar Security | [local_doc_qa_eval_corpus.md chunk 17 ection. ## CASE-027: Frost Gateway 25 Record ID: CASE-027. Project codename: Frost Gateway 25. Owner team: Blue Harbor. Approval path: fina... |
| 28 | OWNER-CASE-028 | single-owner | 0.35 | 1 | 0.00 | 1 | 5451 | Orbit Support | [local_doc_qa_eval_corpus.md chunk 18] |
| 29 | OWNER-CASE-029 | single-owner | 0.35 | 1 | 0.00 | 1 | 6750 | Lumen Data | [local_doc_qa_eval_corpus.md chunk 1] |
| 30 | OWNER-CASE-030 | single-owner | 0.35 | 1 | 0.00 | 1 | 6275 | Aster Compliance | [local_doc_qa_eval_corpus.md chunk 19] local_doc_qa_eval_corpus.md chunk 18 |
| 31 | OWNER-CASE-031 | single-owner | 0.35 | 1 | 0.00 | 1 | 6325 | Northstar Ops | [local_doc_qa_eval_corpus.md chunk 21] |
| 32 | OWNER-CASE-032 | single-owner | 0.35 | 1 | 0.00 | 1 | 5647 | Neon Platform | [owner team: Aster Compliance, local_doc_qa_eval_corpus.md chunk 20] |
| 33 | OWNER-CASE-033 | single-owner | 0.35 | 1 | 0.00 | 1 | 4688 | Blue Harbor | [owner_team] [local_doc_qa_eval_corpus.md chunk 21] |
| 34 | OWNER-CASE-034 | single-owner | 0.35 | 1 | 0.00 | 1 | 5073 | Quartz Finance | [owner_team] [local_doc_qa_eval_corpus.md chunk 22] |
| 35 | OWNER-CASE-035 | single-owner | 0.35 | 1 | 0.00 | 1 | 5744 | Cedar Security | [local_doc_qa_eval_corpus.md chunk 22] |
| 36 | OWNER-CASE-036 | single-owner | 0.35 | 1 | 0.00 | 1 | 6877 | Orbit Support | [local_doc_qa_eval_corpus.md chunk 23] |
| 37 | OWNER-CASE-037 | single-owner | 0.35 | 1 | 0.00 | 1 | 23509 | Lumen Data | [1] local_doc_qa_eval_corpus.md chunk 23 the exact owner, approval path, deadline, region, budget, retention window, and severity values written in this section. ## CASE-037: Fr... |
| 38 | OWNER-CASE-038 | single-owner | 0.35 | 1 | 0.00 | 1 | 6407 | Aster Compliance | [local_doc_qa_eval_corpus.md chunk 24] |
| 39 | OWNER-CASE-039 | single-owner | 0.35 | 1 | 0.00 | 1 | 5261 | Northstar Ops | [Lumen Data]([notes.md]) |
| 40 | OWNER-CASE-040 | single-owner | 0.35 | 1 | 0.00 | 1 | 5701 | Neon Platform | [local_doc_qa_eval_corpus.md chunk 1] |
| 41 | DEADLINE-CASE-011 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5869 | 2026-03-07 | The exact deadline for CASE-011 is 2026-01-01. |
| 42 | DEADLINE-CASE-012 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6075 | 2026-04-10 | The exact deadline for CASE-012 is 2026-02-04. |
| 43 | DEADLINE-CASE-013 | single-deadline | 0.25 | 1 | 0.00 | 0 | 7317 | 2026-05-13 | The exact deadline for CASE-013 is 2026-04-10. |
| 44 | DEADLINE-CASE-014 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5118 | 2026-06-16 | The exact deadline for CASE-014 is 2026-04-10. |
| 45 | DEADLINE-CASE-015 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5807 | 2026-07-19 | The exact deadline for CASE-015 is 2026-05-13. |
| 46 | DEADLINE-CASE-016 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6095 | 2026-08-22 | The exact deadline for CASE-016 is 2026-07-19. |
| 47 | DEADLINE-CASE-017 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6111 | 2026-09-25 | The exact deadline for CASE-017 is 2026-07-19. |
| 48 | DEADLINE-CASE-018 | single-deadline | 0.25 | 1 | 0.00 | 0 | 8529 | 2026-01-01 | The exact deadline for CASE-018 is 2026-07-19. |
| 49 | DEADLINE-CASE-019 | single-deadline | 0.25 | 1 | 0.00 | 0 | 7008 | 2026-02-04 | The exact deadline for CASE-019 is 2026-09-25. |
| 50 | DEADLINE-CASE-020 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5946 | 2026-03-07 | The exact deadline for CASE-020 is 2026-01-01. |
| 51 | DEADLINE-CASE-021 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6150 | 2026-04-10 | The exact deadline for CASE-021 is 2026-03-07. |
| 52 | DEADLINE-CASE-022 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6018 | 2026-05-13 | The exact deadline for CASE-022 is 2026-03-07. |
| 53 | DEADLINE-CASE-023 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6408 | 2026-06-16 | The exact deadline for CASE-023 is 2026-05-13. |
| 54 | DEADLINE-CASE-024 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6152 | 2026-07-19 | The exact deadline for CASE-024 is 2026-05-13. |
| 55 | DEADLINE-CASE-025 | single-deadline | 0.25 | 1 | 0.00 | 0 | 4994 | 2026-08-22 | The exact deadline for CASE-025 is 2026-07-19. |
| 56 | DEADLINE-CASE-026 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6369 | 2026-09-25 | The exact deadline for CASE-026 is 2026-08-22. |
| 57 | DEADLINE-CASE-027 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6616 | 2026-01-01 | The exact deadline for CASE-027 is 2026-08-22. |
| 58 | DEADLINE-CASE-028 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6309 | 2026-02-04 | The exact deadline for CASE-028 is 2026-09-25. |
| 59 | DEADLINE-CASE-029 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5945 | 2026-03-07 | The exact deadline for CASE-029 is 2026-09-25. |
| 60 | DEADLINE-CASE-030 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5930 | 2026-04-10 | The exact deadline for CASE-030 is 2026-02-04. |
| 61 | DEADLINE-CASE-031 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6614 | 2026-05-13 | The exact deadline for CASE-031 is 2026-04-10. |
| 62 | DEADLINE-CASE-032 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6893 | 2026-06-16 | The exact deadline for CASE-032 is 2026-04-10. |
| 63 | DEADLINE-CASE-033 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5538 | 2026-07-19 | The exact deadline for CASE-033 is 2026-05-13. |
| 64 | DEADLINE-CASE-034 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5900 | 2026-08-22 | The exact deadline for CASE-034 is 2026-07-19. |
| 65 | DEADLINE-CASE-035 | single-deadline | 0.35 | 1 | 0.00 | 1 | 4539 | 2026-09-25 | [1] local_doc_qa_eval_corpus.md chunk 22 |
| 66 | DEADLINE-CASE-036 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5798 | 2026-01-01 | The exact deadline for CASE-036 is 2026-07-19. |
| 67 | DEADLINE-CASE-037 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6185 | 2026-02-04 | The exact deadline for CASE-037 is 2026-09-25. |
| 68 | DEADLINE-CASE-038 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6297 | 2026-03-07 | The exact deadline for CASE-038 is 2026-02-04. |
| 69 | DEADLINE-CASE-039 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6334 | 2026-04-10 | The exact deadline for CASE-039 is 2026-01-01. |
| 70 | DEADLINE-CASE-040 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6077 | 2026-05-13 | The exact deadline for CASE-040 is 2026-03-07. |
| 71 | DEADLINE-CASE-041 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6651 | 2026-06-16 | The exact deadline for CASE-041 is 2026-05-13. |
| 72 | DEADLINE-CASE-042 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6168 | 2026-07-19 | The exact deadline for CASE-042 is 2026-05-13. |
| 73 | DEADLINE-CASE-043 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5600 | 2026-08-22 | The exact deadline for CASE-043 is 2026-06-16. |
| 74 | DEADLINE-CASE-044 | single-deadline | 0.25 | 1 | 0.00 | 0 | 7683 | 2026-09-25 | The exact deadline for CASE-044 is 2026-08-22. |
| 75 | DEADLINE-CASE-045 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5245 | 2026-01-01 | The exact deadline for CASE-045 is 2026-08-22. |
| 76 | DEADLINE-CASE-046 | single-deadline | 0.25 | 1 | 0.00 | 0 | 7155 | 2026-02-04 | The exact deadline for CASE-046 is 2026-09-25. |
| 77 | DEADLINE-CASE-047 | single-deadline | 0.25 | 1 | 0.00 | 0 | 6749 | 2026-03-07 | The exact deadline for CASE-047 is 2026-02-04. |
| 78 | DEADLINE-CASE-048 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5523 | 2026-04-10 | The exact deadline for CASE-048 is 2026-02-04. |
| 79 | DEADLINE-CASE-049 | single-deadline | 0.35 | 1 | 0.00 | 1 | 25356 | 2026-05-13 | [1] local_doc_qa_eval_corpus.md chunk 31 l. Primary region: Berlin. Deadline: 2026-03-07. Budget cap: $3409. Retention window: 19 days. Severity class: critical. Operational not... |
| 80 | DEADLINE-CASE-050 | single-deadline | 0.25 | 1 | 0.00 | 0 | 5788 | 2026-06-16 | The exact deadline for CASE-050 is 2026-03-07. |
| 81 | BUDGET-CASE-021 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4619 | 2187 | $2140 |
| 82 | BUDGET-CASE-022 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4631 | 2234 | $2140 |
| 83 | BUDGET-CASE-023 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5527 | 2281 | 2234 |
| 84 | BUDGET-CASE-024 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5988 | 2328 | $2234. |
| 85 | BUDGET-CASE-025 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4465 | 2375 | $2328. |
| 86 | BUDGET-CASE-026 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4369 | 2422 | $2375 |
| 87 | BUDGET-CASE-027 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5065 | 2469 | $2375 |
| 88 | BUDGET-CASE-028 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4178 | 2516 | $2469 |
| 89 | BUDGET-CASE-029 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5992 | 2563 | $2422 |
| 90 | BUDGET-CASE-030 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5427 | 2610 | $2469 |
| 91 | BUDGET-CASE-031 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5598 | 2657 | $2610 |
| 92 | BUDGET-CASE-032 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 3993 | 2704 | $2610 |
| 93 | BUDGET-CASE-033 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4894 | 2751 | $2704 |
| 94 | BUDGET-CASE-034 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4609 | 2798 | $2751 |
| 95 | BUDGET-CASE-035 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4323 | 2845 | $2657 |
| 96 | BUDGET-CASE-036 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 6235 | 2892 | 2845 |
| 97 | BUDGET-CASE-037 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5541 | 2939 | $2845 |
| 98 | BUDGET-CASE-038 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4827 | 2986 | 2892 |
| 99 | BUDGET-CASE-039 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4039 | 3033 | $293 |
| 100 | BUDGET-CASE-040 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4727 | 3080 | $293 |
| 101 | BUDGET-CASE-041 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5250 | 3127 | $3080 |
| 102 | BUDGET-CASE-042 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5674 | 3174 | $3080 |
| 103 | BUDGET-CASE-043 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4939 | 3221 | $3127 |
| 104 | BUDGET-CASE-044 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 4296 | 3268 | $3221 |
| 105 | BUDGET-CASE-045 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5173 | 3315 | $3221 |
| 106 | BUDGET-CASE-046 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 3953 | 3362 | $3268 |
| 107 | BUDGET-CASE-047 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 5323 | 3409 | $3362 |
| 108 | BUDGET-CASE-048 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 6250 | 3456 | $3362 |
| 109 | BUDGET-CASE-049 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 6185 | 3503 | $3456 |
| 110 | BUDGET-CASE-050 | numeric-budget | 0.25 | 1 | 0.00 | 0 | 3927 | 3550 | $3409 |
| 111 | MULTI-CASE-001 | multi-field | 0.90 | 1 | 1.00 | 0 | 5874 | finance director approval; Seattle | CASE-001. Project codename: Beacon Ledger 1. Owner team: Blue Harbor. Approval path: finance director approval. Primary region: Seattle. |
| 112 | MULTI-CASE-003 | multi-field | 0.25 | 1 | 0.00 | 0 | 5999 | legal review; Singapore | CASE-003. Owner team: Blue Harbor. Approval path: finance director approval. Primary region: Seattle. |
| 113 | MULTI-CASE-005 | multi-field | 0.25 | 1 | 0.00 | 0 | 6407 | incident commander approval; Berlin | primary region: Toronto. approval path: data steward approval. |
| 114 | MULTI-CASE-007 | multi-field | 0.25 | 1 | 0.00 | 0 | 6060 | finance director approval; Seattle | Primary region: Bengaluru. Approval path: manager approval. |
| 115 | MULTI-CASE-009 | multi-field | 1.00 | 1 | 1.00 | 1 | 6580 | legal review; Singapore | [local_doc_qa_eval_corpus.md chunk 22] archive 33 Record ID: CASE-033. Project codename: Delta Archive 33. Owner team: Blue Harbor. Approval path: legal review. Primary region: ... |
| 116 | MULTI-CASE-011 | multi-field | 0.35 | 1 | 0.00 | 1 | 4801 | incident commander approval; Berlin | [approval_path: legal review, primary_region: Singapore] |
| 117 | MULTI-CASE-013 | multi-field | 0.25 | 1 | 0.00 | 0 | 4634 | finance director approval; Seattle | Primary region: Bengaluru. Approval path: manager approval. |
| 118 | MULTI-CASE-015 | multi-field | 0.25 | 1 | 0.00 | 0 | 6589 | legal review; Singapore | Primary region: Seattle. Approval path: security review. |
| 119 | MULTI-CASE-017 | multi-field | 0.25 | 1 | 0.00 | 0 | 6295 | incident commander approval; Berlin | Primary region: Seattle. Approval path: legal review. |
| 120 | MULTI-CASE-019 | multi-field | 1.00 | 1 | 1.00 | 1 | 27520 | finance director approval; Seattle | [1] local_doc_qa_eval_corpus.md chunk 12 the exact owner, approval path, deadline, region, budget, retention window, and severity values written in this section. ## CASE-019: He... |
| 121 | MULTI-CASE-021 | multi-field | 0.25 | 1 | 0.00 | 0 | 5564 | legal review; Singapore | Primary region: Dublin. Approval path: security review. |
| 122 | MULTI-CASE-023 | multi-field | 0.25 | 1 | 0.00 | 0 | 4635 | incident commander approval; Berlin | **Approval path:** data steward approval **Primary region:** Toronto |
| 123 | MULTI-CASE-025 | multi-field | 0.25 | 1 | 0.00 | 0 | 4553 | finance director approval; Seattle | Primary region: Berlin. Approval path: manager approval. |
| 124 | MULTI-CASE-027 | multi-field | 0.25 | 1 | 0.00 | 0 | 5151 | legal review; Singapore | primary region: Seattle. approval path: finance director approval. |
| 125 | MULTI-CASE-029 | multi-field | 0.25 | 1 | 0.00 | 0 | 7444 | incident commander approval; Berlin | Primary region: Dublin. Approval path: legal review. |
| 126 | MULTI-CASE-031 | multi-field | 0.25 | 1 | 0.00 | 0 | 6074 | finance director approval; Seattle | Primary region: Dublin. Approval path: legal review. |
| 127 | MULTI-CASE-033 | multi-field | 0.25 | 1 | 0.00 | 0 | 4675 | legal review; Singapore | Primary region: Seattle. Approval path: security review. |
| 128 | MULTI-CASE-035 | multi-field | 0.35 | 1 | 0.00 | 1 | 14019 | incident commander approval; Berlin | [1] local_doc_qa_eval_corpus.md chunk 22 archive 33 Record ID: CASE-033. Project codename: Delta Archive 33. Owner team: Blue Harbor. Approval path: legal review. Primary region... |
| 129 | MULTI-CASE-037 | multi-field | 0.25 | 1 | 0.00 | 0 | 12249 | finance director approval; Seattle | The exact owner, approval path, deadline, region, budget, retention window, and severity values written in this section for CASE-037 are as follows: - Owner team: Cedar Security... |
| 130 | MULTI-CASE-039 | multi-field | 0.25 | 1 | 0.00 | 0 | 4922 | legal review; Singapore | Primary region: Seattle. Approval path: finance director approval. |
| 131 | MULTI-CASE-041 | multi-field | 0.25 | 1 | 0.00 | 0 | 5125 | incident commander approval; Berlin | The approval path for CASE-041 is data steward approval, and the primary region is Toronto. |
| 132 | MULTI-CASE-043 | multi-field | 0.25 | 1 | 0.00 | 0 | 5117 | finance director approval; Seattle | Primary region: Bengaluru. Approval path: manager approval. |
| 133 | MULTI-CASE-045 | multi-field | 0.25 | 1 | 0.00 | 0 | 5379 | legal review; Singapore | Primary region: Seattle. Approval path: finance director approval. |
| 134 | MULTI-CASE-047 | multi-field | 0.25 | 1 | 0.00 | 0 | 5898 | incident commander approval; Berlin | Primary region: Toronto. Approval path: data steward approval. |
| 135 | MULTI-CASE-049 | multi-field | 0.25 | 1 | 0.00 | 0 | 5890 | finance director approval; Seattle | Primary region: Singapore. Approval path: legal review. |
| 136 | RETENTION-CASE-004 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 4492 | 18; days | 29 days |
| 137 | RETENTION-CASE-009 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 4047 | 23; days | 21 days |
| 138 | RETENTION-CASE-014 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 4882 | 28; days | 26 days |
| 139 | RETENTION-CASE-019 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 4939 | 33; days | 31 days |
| 140 | RETENTION-CASE-024 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 5792 | 17; days | 15 days |
| 141 | RETENTION-CASE-029 | numeric-retention | 0.25 | 1 | 0.00 | 0 | 5099 | 22; days | 2026-09-25 |
| 142 | RETENTION-CASE-034 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 4571 | 27; days | 26 days |
| 143 | RETENTION-CASE-039 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 4559 | 32; days | 29 days |
| 144 | RETENTION-CASE-044 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 4150 | 16; days | 35 days |
| 145 | RETENTION-CASE-049 | numeric-retention | 0.57 | 1 | 0.50 | 0 | 5206 | 21; days | 18 days |
| 146 | MISSING-CASE-090 | unanswerable | 0.35 | 1 | 0.00 | 1 | 5410 | missing | [notes.md] |
| 147 | MISSING-CASE-091 | unanswerable | 0.25 | 1 | 0.00 | 0 | 5964 | missing | CASE-091. Owner team: Neon Platform. |
| 148 | MISSING-CASE-092 | unanswerable | 0.25 | 1 | 0.00 | 0 | 4154 | missing | The owner team for CASE-092 is Blue Harbor. |
| 149 | MISSING-CASE-093 | unanswerable | 0.25 | 1 | 0.00 | 0 | 4946 | missing | The owner team for CASE-093 is Blue Harbor. |
| 150 | MISSING-CASE-094 | unanswerable | 0.25 | 1 | 0.00 | 0 | 5101 | missing | CASE-094. Owner team: Neon Platform. |
