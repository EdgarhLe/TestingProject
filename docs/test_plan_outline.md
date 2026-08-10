# Test Plan Outline

## 1. Objective

This document defines the overall testing strategy for the Multimedia Retrieval System. It describes what will be tested, how testing will be performed, the testing environment, and the testing timeline.

---

# 2. Scope

The following system components will be tested:

| Component | Description |
|------------|-------------|
| Model | AI retrieval models (KIS, TRAKE, QA) |
| Indexing | Feature extraction and indexing pipeline |
| API | Backend REST API |
| Frontend | User interface and interaction |

---

# 3. Test Types

## Unit Testing

Purpose:

- Verify individual modules work correctly.

Examples:

- Model utilities
- API helper functions
- Indexing utilities

---

## Integration Testing

Purpose:

Verify communication between system components.

Examples:

- Frontend ↔ API
- API ↔ Model
- API ↔ Index

---

## End-to-End Testing

Purpose:

Validate complete user workflows.

Example:

User uploads query

↓

API receives request

↓

Model performs retrieval

↓

Results returned

↓

Frontend displays results

---

# 4. Query-specific Testing

## KIS

Test items

- valid image query
- invalid query
- response format
- response latency

Expected Output

- Top-k results returned
- Valid video_id
- Valid frame_index

---

## TRAKE

Test items

- event retrieval
- multiple candidates
- found=false handling

Expected Output

- Candidate list returned
- Event timestamps valid

---

## Q&A

Test items

- question answering
- answer source
- response format

Expected Output

- Answer generated
- Source information included

---

# 5. Testing Environment

All development and testing activities are performed on Machine 2.

Test Machine: Machine 2

- OS: Windows 11
- Python: 3.11
- FastAPI
- VS Code

Backend

- localhost:8000

Frontend

- localhost:3000

Mock API (if used)

- localhost:8001

# 6. Test Schedule

Testing activities will be aligned with the team's 12-week development plan.

| Period | Activities |
| ------ | ---------- |
| Week 1–2 | Prepare testing environment and test setup |
| Week 3–5 | Unit testing |
| Week 6–8 | Integration testing |
| Week 9–10 | End-to-end testing |
| Week 11 | Regression testing |
| Week 12 | Final validation and test summary |

# 7. Success Criteria

Testing is considered complete when the following criteria are met:

## Unit Testing

- Individual modules and utility functions work correctly.
- Unit tests pass for the tested components.
- Invalid inputs are handled as expected.

## Integration Testing

- Frontend communicates correctly with the API.
- API communicates correctly with the model and indexing components.
- Request and response formats conform to the API contract.
- Integration tests pass for the tested interfaces.

## End-to-End Testing

- Complete user workflows execute successfully from query submission to result display.
- KIS, TRAKE, and Q&A queries return valid results according to the defined API contract.
- No critical errors occur during the complete workflow.

## Overall System

- No critical bugs remain before final validation.
- Test results and relevant documentation are complete.


# 8. Deliverables

- Test Plan Outline
- Test Cases
- Bug Reports
- Test Summary Report