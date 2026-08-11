
Evaluation Metrics

This document defines the official evaluation metrics for the Multimedia Retrieval System based on the AIC 2026 preliminary-round evaluation specification.

The system supports three query types:

Textual Known Item Search (Textual KIS)

Question Answering (Q&A)

Temporal Retrieval and Alignment of Key Events (TRAKE)

1. Evaluation Overview

For each query, a team may submit up to 100 candidate answers.

Each candidate answer is assigned an R-Score, ranging from 0 to 1:

1: completely correct

0: incorrect

intermediate values: partially correct

The final score for a query is calculated from the best R-Score values obtained at different ranking positions.

The official ranking thresholds are:

k ∈ {1, 5, 20, 50, 100}

2. R-Score

2.1 Textual KIS

Response Format

<video_id>, <frame_id>

Correctness Condition

A response is correct if:

video_id matches the ground-truth video.

frame_id is within the ground-truth frame interval [s, e].

Formula

R-Score(ri) = I(vi = GTv ∧ idi ∈ [s, e])

where:

ri: submitted response

vi: submitted video ID

GTv: ground-truth video ID

idi: submitted frame ID

[s, e]: valid ground-truth frame interval

I(.): indicator function returning 1 if the condition is true and 0 otherwise

Example

Ground truth:

video_id = L01_V001
frame range = [500, 510]

Response

R-Score

L01_V001, 505

1

L01_V001, 600

0

L02_V003, 505

0

2.2 Question Answering (Q&A)

Response Format

<video_id>, <frame_id>, <answer>

Correctness Condition

A response is correct if all three conditions are satisfied:

video_id matches the ground-truth video.

frame_id is within the ground-truth frame interval [s, e].

answer matches the ground-truth answer semantically.

Formula

R-Score(ri) = I(vi = GTv ∧ idi ∈ [s, e] ∧ ai = GTa)

where:

vi: submitted video ID

GTv: ground-truth video ID

idi: submitted frame ID

[s, e]: valid ground-truth frame interval

ai: submitted answer

GTa: ground-truth answer

Example

Ground truth:

video_id = L05_V005
frame range = [800, 900]
answer = "màu xanh"

Response

R-Score

L05_V005, 888, màu xanh

1

L05_V005, 888, màu trắng

0

L06_V007, 888, màu xanh

0

2.3 TRAKE

Response Format

<video_id>, <frame_id1>, ..., <frame_idN>

where N is the number of semantic events in the query.

Correctness Condition

The submitted video_id must match the ground-truth video.

If the video is incorrect:

R-Score(ri) = 0

If the video is correct, the score is the proportion of semantic events whose submitted frame falls inside the corresponding ground-truth frame interval.

Formula

When vi = GTv:

R-Score(ri) = (1 / N) × Σ I(id_i,j ∈ [s_j, e_j])

Otherwise:

R-Score(ri) = 0

where:

N: total number of semantic events

id_i,j: submitted frame for event j

[s_j, e_j]: valid frame interval for event j

I(.): indicator function

Example

Ground truth:

Event 1: [95, 105]
Event 2: [145, 155]
Event 3: [195, 205]
Event 4: [245, 255]

Submitted:

L10_V010, 101, 156, 203, 251

Evaluation:

Event 1: 101 ∈ [95, 105]    → Correct
Event 2: 156 ∉ [145, 155]   → Incorrect
Event 3: 203 ∈ [195, 205]   → Correct
Event 4: 251 ∈ [245, 255]   → Correct

Therefore:

R-Score = 3 / 4 = 0.75

3. Ranking Metrics

For each query, up to 100 candidate responses may be submitted.

For each ranking threshold:

k ∈ {1, 5, 20, 50, 100}

the system calculates the best R-Score among the first k responses.

R@k

R@k = max{R-Score(ri) | 1 ≤ i ≤ k}

The official ranking metrics are:

R@1

R@5

R@20

R@50

R@100

4. Final Score

The final score for each query is the arithmetic mean of the five ranking scores:

Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5

Example

Suppose:

R@1   = 0.5
R@5   = 0.8
R@20  = 0.8
R@50  = 0.8
R@100 = 0.8

Then:

Final Score = (0.5 + 0.8 + 0.8 + 0.8 + 0.8) / 5 = 0.74

5. Metric Summary

Query Type

R-Score Condition

Ranking Metrics

Textual KIS

Correct video + frame within [s,e]

R@1, R@5, R@20, R@50, R@100

Q&A

Correct video + frame within [s,e] + correct answer

R@1, R@5, R@20, R@50, R@100

TRAKE

Correct video + proportion of correctly aligned events

R@1, R@5, R@20, R@50, R@100

6. Important Notes

The metrics defined in this document follow the latest AIC 2026 preliminary-round evaluation specification.

The official evaluation uses R-Score and R@k as described above.

The following metrics are not used as the official final evaluation metrics in this specification:

MRR

IoU

EM

F1

These metrics should not be introduced as official evaluation criteria unless a newer official specification explicitly requires them.

The final evaluation is based on:

R@1
R@5
R@20
R@50
R@100

and their average:

Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5