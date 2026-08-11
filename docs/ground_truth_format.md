# Ground Truth and Submission CSV Format

## Purpose

This document defines:
- Internal ground truth JSON format used for evaluation.
- Official submission CSV format for AI Challenge.
- Notes and rules for CSV submission.

## Internal Ground Truth Format

### KIS
```json
{
  "query_id": "p1-1",
  "video_id": "L01_V001",
  "frame_index": 1042
}
```

### TRAKE
```json
{
  "query_id": "p1-5",
  "video_id": "L01_V001",
  "event_index": 0,
  "frame_index": 512
}
```

### Q&A
```json
{
  "query_id": "p1-3",
  "video_id": "L02_V003",
  "frame_index": 2310,
  "answer": "Nguyễn Văn An"
}
```

## Official Submission CSV Format
The official submission format follows the AI Challenge preliminary submission guideline.

### KIS
```text
video_id,frame_id
```

Example:

```text
L00_V000,1234
L00_V055,5555
L01_V028,25380
```

### Q&A
```text
video_id,frame_id,answer
```

Example:

```text
L01_V028,3450,"5"
L02_V011,1200,Nhìn người
L03_V005,2800,"Màu đỏ"
```

### TRAKE
```text
<video_id>, <Frame ID_1>, <Frame ID_2>, ..., <Frame ID_N>
```

Example:

```text
L10_V001,1200,1850,2100,2450
L10_V001,1180,1820,2000,2420
```

## CSV Rules
- Encoding: UTF-8
- Delimiter: comma (,)
- No header row
- One CSV file is generated for each query package.
- Submit all CSV files inside a `submission` folder and compress it into a `.zip` file.

## Evaluation Metrics

### 1. R-Score (Relevant Score per prediction)
- **Textual KIS**: 
  $$R\text{-Score}(r_i) = I(v_i = GT_v \land id_i \in [s, e])$$
  Returns 1 if `video_id` matches and `frame_id` falls within `[s, e]`; otherwise 0.

- **Q&A**:
  $$R\text{-Score}(r_i) = I(v_i = GT_v \land id_i \in [s, e] \land a_i = GT_a)$$
  Returns 1 if `video_id` matches, `frame_id` falls within `[s, e]`, AND `answer` matches semantically (`GT_a`); otherwise 0.

- **TRAKE**:
  $$R\text{-Score}(r_i) = \frac{1}{N} \sum_{j=1}^N I(id_{i,j} \in [s_j, e_j]) \quad \text{if } v_i = GT_v, \text{ else } 0$$
  If correct `video_id`, returns proportion of keyframes matching their corresponding ground truth ranges `[s_j, e_j]`.

### 2. Final Score (per query)
- Up to 100 predictions ($r_1, r_2, \dots, r_{100}$) are submitted per query.
- For each threshold $k \in \{1, 5, 20, 50, 100\}$, calculate $R@k = \max_{1 \le i \le k} \{R\text{-Score}(r_i)\}$.
- Final Score is the average across all 5 thresholds:
  $$\text{Final Score} = \frac{1}{5} \sum_{k \in \{1, 5, 20, 50, 100\}} R@k$$

## Points to Confirm
- Confirm whether frame index starts from 0 or 1.
- Confirm the evaluation method for Q&A answers (semantic match vs exact match).
- Confirm the expected ordering of frames in TRAKE submissions.
- Confirm whether the submission format will change in future competition rounds.

## References
- AI Challenge Preliminary Submission Guideline
- docs/api_contract_v1.md
