## Метрики по режимам

| mode   | name            |   n_base |   accuracy |   ev_precision |   ev_recall |   ev_f1 |   cev_recall |   both_precision_all |   both_recall_all |   insuf_precision_all |   insuf_recall_all |   confident_when_insufficient |   valid_first_try |   invalid_final |   prompt_tokens_mean |   completion_tokens_mean |   latency_median_s |   latency_mean_s |   confidence_mean |   accuracy_all_runs |   n_all_runs |
|:-------|:----------------|---------:|-----------:|---------------:|------------:|--------:|-------------:|---------------------:|------------------:|----------------------:|-------------------:|------------------------------:|------------------:|----------------:|---------------------:|-------------------------:|-------------------:|-----------------:|------------------:|--------------------:|-------------:|
| A      | Полный контекст |       12 |        0.5 |          0.265 |       0.294 |   0.262 |        0     |              nan     |             0     |                  0.6  |              0.375 |                         0.625 |                 1 |               0 |                15865 |                      163 |               8.41 |            15.19 |             0.917 |               0.325 |           40 |
| B      | BM25 top-15     |       12 |        0.5 |          0.368 |       0.436 |   0.37  |        0     |              nan     |             0     |                  0.75 |              0.375 |                         0.625 |                 1 |               0 |                 1311 |                      146 |               8.11 |             8.17 |             0.95  |               0.325 |           40 |
| C      | p-adic контекст |       12 |        0.5 |          0.459 |       0.525 |   0.468 |        0.125 |                0.714 |             0.357 |                  1    |              0.125 |                         0.625 |                 1 |               0 |                 1500 |                      142 |               8.56 |             8.6  |             0.946 |               0.4   |           40 |

## Устойчивость

| mode   |   perm_changed_share |   perm_accuracy |   perm_ev_jaccard |   remove_expected_share |   remove_changed_share |   contra_both_share |   contra_visible |   contra_both_share_visible |   contra_changed_share |   base_accuracy |
|:-------|---------------------:|----------------:|------------------:|------------------------:|-----------------------:|--------------------:|-----------------:|----------------------------:|-----------------------:|----------------:|
| A      |                0.083 |           0.583 |             0.473 |                    0    |                   0    |                0    |               12 |                         0   |                  0.083 |             0.5 |
| B      |                0.083 |           0.583 |             0.635 |                    0    |                   0    |                0    |               10 |                         0   |                  0     |             0.5 |
| C      |                0     |           0.5   |             1     |                    0.25 |                   0.25 |                0.25 |               10 |                         0.3 |                  0.083 |             0.5 |

## Калибровка

| mode   | bin          |   n |   confidence_mean |   accuracy |
|:-------|:-------------|----:|------------------:|-----------:|
| A      | [0.00; 0.60) |   5 |             0.5   |      0.6   |
| A      | [0.60; 0.80) |   0 |           nan     |    nan     |
| A      | [0.80; 0.95) |   1 |             0.85  |      0     |
| A      | [0.95; 1.00] |  34 |             0.968 |      0.294 |
| A      | ECE          |  40 |           nan     |      0.606 |
| B      | [0.00; 0.60) |   1 |             0.5   |      1     |
| B      | [0.60; 0.80) |   3 |             0.7   |      0.667 |
| B      | [0.80; 0.95) |   0 |           nan     |    nan     |
| B      | [0.95; 1.00] |  36 |             0.975 |      0.278 |
| B      | ECE          |  40 |           nan     |      0.643 |
| C      | [0.00; 0.60) |   0 |           nan     |    nan     |
| C      | [0.60; 0.80) |   2 |             0.7   |      1     |
| C      | [0.80; 0.95) |   6 |             0.8   |      0.667 |
| C      | [0.95; 1.00] |  32 |             0.972 |      0.312 |
| C      | ECE          |  40 |           nan     |      0.563 |

## Плоский и p-adic контекст

| question_id   | question_type   | gold         |   confidence_B |   confidence_C |   ev_f1_B |   ev_f1_C | status_B     | status_C   | same_status   |
|:--------------|:----------------|:-------------|---------------:|---------------:|----------:|----------:|:-------------|:-----------|:--------------|
| Q01           | direct          | entailed     |           1    |           1    |  1        |  1        | entailed     | entailed   | True          |
| Q02           | direct          | entailed     |           1    |           1    |  0.285714 |  1        | entailed     | entailed   | True          |
| Q03           | multihop        | entailed     |           0.95 |           0.95 |  0.285714 |  0.285714 | entailed     | entailed   | True          |
| Q04           | multihop        | contradicted |           0.95 |           0.95 |  0.285714 |  0        | entailed     | entailed   | True          |
| Q05           | temporal        | entailed     |           1    |           1    |  1        |  0.75     | entailed     | entailed   | True          |
| Q06           | temporal        | contradicted |           1    |           1    |  0.8      |  0.571429 | entailed     | entailed   | True          |
| Q07           | rule_exception  | entailed     |           0.95 |           0.95 |  0.444444 |  0.533333 | entailed     | entailed   | True          |
| Q08           | rule_exception  | contradicted |           1    |           0.95 |  0        |  0.666667 | entailed     | entailed   | True          |
| Q09           | contradiction   | contradicted |           0.95 |           0.95 |  0        |  0        | entailed     | entailed   | True          |
| Q10           | contradiction   | both         |           0.95 |           0.8  |  0.333333 |  0.666667 | entailed     | both       | False         |
| Q11           | insufficient    | insufficient |           0.95 |           1    |  0        |  0        | entailed     | entailed   | True          |
| Q12           | insufficient    | insufficient |           0.7  |           0.8  |  0        |  0.142857 | insufficient | both       | False         |

## Поиск BM25

| question_id   | type           | gold_ids                | missed   |   recall_at_15 | top15                                                       |
|:--------------|:---------------|:------------------------|:---------|---------------:|:------------------------------------------------------------|
| Q01           | direct         | C17                     |          |          1     | C17 C40 C13 C18 C20 C08 C03 C30 C39 C07 C38 C15 C29 C28 C06 |
| Q02           | direct         | C05                     |          |          1     | C05 C11 C36 C01 C19 C21 C30 C02 C04 C40 C07 C25 C26 C08 C17 |
| Q03           | multihop       | C05 C21 C20             | C21      |          0.667 | C36 C22 C01 C23 C20 C05 C11 C02 C18 C34 C07 C06 C04 C13 C24 |
| Q04           | multihop       | C05 C18 C19             | C19      |          0.667 | C36 C01 C18 C25 C11 C05 C30 C02 C22 C23 C20 C09 C10 C04 C12 |
| Q05           | temporal       | C01 C03 C23             | C03      |          0.667 | C23 C36 C01 C22 C20 C11 C02 C07 C33 C05 C40 C25 C26 C18 C34 |
| Q06           | temporal       | C02 C03 C20             | C03      |          0.667 | C02 C04 C20 C22 C23 C06 C07 C15 C18 C39 C38 C08 C17 C11 C34 |
| Q07           | rule_exception | C28 C33 C20 C31 C32 C13 | C32 C13  |          0.667 | C20 C33 C23 C28 C22 C35 C34 C06 C29 C30 C31 C08 C05 C17 C36 |
| Q08           | rule_exception | C29 C07 C35 C34         | C29 C35  |          0.5   | C23 C22 C20 C33 C13 C15 C37 C09 C07 C27 C34 C14 C24 C25 C18 |
| Q09           | contradiction  | C25 C09 C10 C12         |          |          1     | C12 C25 C18 C24 C02 C09 C26 C36 C17 C10 C11 C37 C08 C15 C16 |
| Q10           | contradiction  | C13 C27 C08 C16         |          |          1     | C08 C13 C27 C18 C20 C33 C17 C16 C22 C23 C29 C28 C14 C05 C40 |
| Q11           | insufficient   | C04                     |          |          1     | C02 C11 C04 C36 C01 C05 C40 C07 C25 C26 C03 C06 C08 C09 C10 |
| Q12           | insufficient   | C05 C21 C14             | C21      |          0.667 | C14 C08 C17 C35 C18 C20 C27 C33 C13 C37 C15 C05 C36 C01 C22 |

