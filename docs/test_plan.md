# Finalized Test Plan

**Project:** SIU_Collective - VL-JEPA Multimedia Retrieval System (AIC 2026)  
**Document Path:** `docs/test_plan.md`  
**Reference Dependencies:** `docs/api_contract_v1.md`, `configs/model.yaml`, `README.md`, Issue #31, Issue #16, Issue #20  

---

## 1. Objective

This document defines the finalized testing strategy for the VL-JEPA Multimedia Retrieval System. It establishes **what to test**, **how to test** (testing framework, test data, and execution procedures), and the **success criteria** for every component in the system based strictly on the system design specifications.

The primary goal is to ensure system reliability across all modules (X-Encoder, Predictor, Y-Encoder, ASR, OCR, Semantic Segmentation, FAISS Indexing, API Gateway, and Frontend) across unit tests, integration tests, and end-to-end competition mode execution.

---

## 2. Scope & Architecture Coverage

The system follows the VL-JEPA architecture featuring a **1536-dimensional shared embedding space** (defined in `configs/model.yaml`):
- **Visual Branch:** Image/Frame $\rightarrow$ X-Encoder (V-JEPA 2 ViT-L, output dim 1024) $\rightarrow$ Predictor (Qwen2.5 upper layers) $\rightarrow$ **1536-dim shared embedding**.
- **Text/Query Branch:** Text $\rightarrow$ Y-Encoder (BGE-M3) $\rightarrow$ **1536-dim shared embedding**.
- **Indexing & Search:** FAISS HNSW coarse & fine indexes operate on the **1536-dim shared embedding space** (NOT raw X-Encoder output alone).

| Component Layer | Covered Modules / Endpoints | Testing Scope |
| :--- | :--- | :--- |
| **Unit Level** | X-Encoder, Predictor, Y-Encoder, Whisper ASR, PaddleOCR, Semantic Segmentation, FAISS Indexing, API Schemas | Function correctness, tensor shapes, error handling, isolated boundary inputs |
| **Integration Level** | Frontend ↔ API Gateway, API ↔ VL-JEPA Pipeline (1536-dim shared space), API ↔ FAISS Index | Contract V1 compliance, data flow across KIS, TRAKE, and Q&A pipelines |
| **E2E / Competition** | Batch Processing, Timeline Navigation, Competition Mode Engine | Full workflow execution, batch query handling, VRAM ($\le 16\text{ GB}$) & memory leak checks |

---

## 3. Comprehensive Test Plan

### 3.1 Unit Testing Section

Unit tests verify individual modules, functions, and data transformations in complete isolation using `pytest`.

```
tests/unit/
├── test_x_encoder.py       # X-Encoder (V-JEPA 2 ViT-L) unit tests
├── test_predictor.py       # Predictor (Qwen2.5 upper layers) unit tests
├── test_y_encoder.py       # Y-Encoder (BGE-M3) unit tests
├── test_asr.py             # Whisper ASR module unit tests
├── test_ocr.py             # PaddleOCR module unit tests
├── test_segmentation.py    # Semantic segmentation (sliding window + clustering) tests
├── test_indexing.py        # FAISS 1536-dim index creation, insert, and search tests
└── test_api_schemas.py     # Pydantic request/response schema validation tests
```

#### Module Unit Test Specifications

| Module | What to Test | How to Test (Procedure & Tools) | Success Criteria / Expected Output |
| :--- | :--- | :--- | :--- |
| **X-Encoder** | Visual feature extraction (`model/x_encoder/`). Note: Unit test setup already exists in `tests/unit/x_encoder_test.py`. | Run `pytest tests/unit/x_encoder_test.py`. Pass synthetic image tensors of shape `(3, 256, 256)` (matching design specification of `3,256,256`) and invalid non-image inputs. | • Returns normalized float32 feature tensor of shape `(1024,)`<br>• Accepts `(3, 256, 256)` input size according to model spec<br>• Model remains frozen (`frozen: true`) without gradient updates<br>• Invalid input raises `ValueError` |
| **Predictor** | Predictor embedding generation (`model/predictor/`) | Run `pytest tests/unit/test_predictor.py`. Input X-Encoder output (1024-dim) with bidirectional attention mask. | • Outputs predicted embedding tensor in **1536-dim shared embedding space** (`projection_dim: 1536`) |
| **Y-Encoder** | Text/query embedding generation (`model/y_encoder/`) | Run `pytest tests/unit/test_y_encoder.py` with text query strings. | • Outputs query embedding tensor in **1536-dim shared embedding space** (`projection_dim: 1536`) |
| **ASR (Whisper)** | Audio transcript extraction with timestamps (`indexing/asr/`) | Run `pytest tests/unit/test_asr.py`. Input 5s sample audio `.wav` files (speech vs. silent/noise). | • Returns correct transcript string with timestamp metadata<br>• Silent audio returns empty string `""` without crash<br>• Latency < 2.0s per 5s segment |
| **OCR (PaddleOCR)** | On-screen text detection (`indexing/ocr/`) | Run `pytest tests/unit/test_ocr.py` passing synthetic frames with embedded text vs textless frames. | • Returns list of bounding boxes + text strings<br>• Textless frame returns empty list `[]` |
| **Semantic Segmentation** | Video semantic segmentation (`indexing/segmentation/`) using sliding window + clustering | Run `pytest tests/unit/test_segmentation.py` passing frame feature sequences through sliding window feature extraction and clustering pipeline. | • Generates semantic segment clusters with valid `start_frame` and `end_frame` boundaries<br>• Correctly groups semantically similar consecutive frames without hardcoded fixed-second splits |
| **FAISS Indexing** | HNSW coarse + fine index operations on **1536-dim shared space** (`indexing/indexes/`) | Run `pytest tests/unit/test_indexing.py`. Insert 500 mock **1536-dim** vectors, execute `index.search(query_1536d, k=5)`. | • Operates on **1536-dim** vectors matching shared embedding space<br>• Returns top-5 indices and distance scores<br>• Distance scores contain no `NaN` or `Inf` |
| **API Schemas & Endpoints** | Pydantic Request/Response validation (`api/schemas/`) | Run `pytest tests/unit/test_api_schemas.py`. Test valid JSON vs missing fields (`top_k`, `query_id`). | • Valid payload passes schema validation<br>• Missing/invalid field raises HTTP `422 Unprocessable Entity` according to Contract v1 |

---

### 3.2 Integration Testing Section

Integration tests verify component communication and pipeline data flow from query input to response output according to **API Contract v1** and the **1536-dim VL-JEPA shared embedding space**.

#### 1. KIS Integration Flow (Known-Item Search)
* **What to test:** Query submission $\rightarrow$ Visual Branch (X-Encoder $\rightarrow$ Predictor to 1536-dim) / Text Branch (Y-Encoder BGE-M3 to 1536-dim) $\rightarrow$ FAISS coarse & fine search on **1536-dim shared embedding space** $\rightarrow$ Stride-based re-scoring $\rightarrow$ API response. *(Note: FAISS search is NOT performed on raw X-Encoder output alone, but on the 1536-dim shared space).*
* **How to test:** 
  1. Execute `pytest tests/integration/test_kis_flow.py` using `fastapi.testclient.TestClient`.
  2. Send `POST /query/kis` with payload `{"query_image": "...", "top_k": 5}`.
  3. Validate response structure and query database against ground truth.
* **Success Criteria:**
  * HTTP status code `200 OK`.
  * Response body contains `video_id`, `frame_index`, `timestamp_seconds`, `score`.
  * Metric check: **Recall@5** evaluated on benchmark dataset.

#### 2. TRAKE Integration Flow (Temporal Retrieval of Keyframes)
* **What to test:** Multi-event query breakdown $\rightarrow$ Y-Encoder vectorization to 1536-dim $\rightarrow$ FAISS fine index scan $\rightarrow$ Temporal constraint filter $\rightarrow$ Visual Delta calculation $\rightarrow$ Target frame range response.
* **How to test:**
  1. Execute `pytest tests/integration/test_trake_flow.py`.
  2. Send `POST /query/trake` with sequence of event descriptions.
  3. Test both valid match cases and no-match cases.
* **Success Criteria:**
  * HTTP status code `200 OK`.
  * Returns list of candidate videos with `description`, `timestamp_seconds`, `video_fps`.
  * When no event matches, correctly sets `found: false` in accordance with API Contract v1.

#### 3. Q&A Integration Flow (Question Answering)
* **What to test:** Question analysis $\rightarrow$ Hybrid search (FAISS 1536-dim + BM25 ASR/OCR) $\rightarrow$ Light Y-Decoder / LLM Qwen2.5 / PaddleOCR branch $\rightarrow$ Final answer + source attribution.
* **How to test:**
  1. Execute `pytest tests/integration/test_qa_flow.py`.
  2. Send `POST /query/qa` with text questions targeting OCR vs speech content.
* **Success Criteria:**
  * HTTP status code `200 OK`.
  * Response contains `answer` and `answer_source` (`"OCR"` or `"LLM"`).
  * Evaluated metric: **Exact Match (EM)** or **Word F1-Score** on Q&A benchmark.

---

### 3.3 End-to-End (E2E) & Competition Mode Testing Section

E2E testing validates full system integration from Frontend UI user interactions down to backend batch execution in Competition Mode.

#### 1. E2E Client-Server Interaction
* **What to test:** User action on Frontend UI $\rightarrow$ API Gateway forwarding $\rightarrow$ Async Batch Queue execution $\rightarrow$ Submission CSV generation $\rightarrow$ Video Player Tracker sync.
* **How to test:**
  1. Launch Backend API (`uvicorn api.main:app --port 8000`) and Mock/FE Server.
  2. Load competition query file pack `.txt` via Frontend interface.
  3. Perform search and click result items to trigger timeline navigation.
* **Success Criteria:**
  * Frontend renders video list and syncs Video Player Tracker accurately down to millisecond granularity.
  * System generates valid `submission.csv` matching organiser specifications.

#### 2. Competition Mode Stress & Resource Testing
* **What to test:** Batch endpoints (`POST /query/kis/batch`, `POST /query/trake/batch`, `POST /query/qa/batch`) handling 50 concurrent query files.
* **How to test:**
  1. Run automated load script `python scripts/stress_test_batch.py --concurrency 50`.
  2. Monitor GPU memory usage via `nvidia-smi` and system RAM via `psutil`.
* **Success Criteria:**
  * Zero Memory Leak during prolonged batch execution.
  * Total peak VRAM consumption **$\le 16\text{ GB}$** on Machine 2.
  * Batch response time stays within latency budget (< 500ms per query).

---

## 4. Test Schedule & Phased Milestones

Testing activities are aligned with the 12-week project roadmap:

| Timeline | Phase / Milestone | Key Testing Focus & Deliverables |
| :--- | :--- | :--- |
| **Week 1–2** | Baseline Setup | Define evaluation metrics (`evaluation_metrics.md`) & finalize `api_contract_v1.md`. |
| **Week 3** | BE/FE Independent Start | Run Unit Tests for data preprocessing & offline indexing (1–5 videos). Frontend tests UI with Mock API Server (Contract V1). |
| **Week 4** | Stage A Pre-training | Unit Test Whisper ASR (30–50 videos). Integration test for API Gateway data payload serialization. |
| **Week 5** | Stage B (Checkpoint A) | Trigger Pipeline KIS Integration Test. Measure and report **Recall@5** under varying stride rescoring configurations. |
| **Week 6** | Stage C (Checkpoint B) | Trigger Pipeline Trake Integration Test. Enforce strict temporal constraint filtering & Visual Delta sensitivity. |
| **Week 7** | Stage D (Checkpoint C) | Trigger Pipeline Q&A Integration Test. Verify auto-branching between OCR exact match and LLM Qwen2.5 output (**Word F1-Score**). |
| **Week 8** | SFT Phase (Checkpoint D) | Run Full E2E Competition Mode on Blind Validation set (50 queries). Validate VRAM $\le 16\text{ GB}$ & memory leak prevention. |
| **Week 9–12**| Mock Competitions & Optimization | Continuous stress testing, latency optimization, final submission validation (`submission.csv`). |

---

## 5. Summary of Deliverables & Success Criteria Checklist

To mark Issue #31 as **COMPLETE**, the following criteria must be satisfied:

- [x] **Unit Testing:** All core modules (X-Encoder, Predictor, Y-Encoder, ASR, OCR, Semantic Segmentation, FAISS 1536-dim Indexing, API Schemas) covered with `pytest`.
- [x] **Integration Testing:** KIS, TRAKE, and Q&A pipelines pass contract validation against `docs/api_contract_v1.md` using the 1536-dim shared embedding space.
- [x] **E2E & Stress Testing:** Batch execution verified with memory limit $\le 16\text{ GB}$ VRAM and valid `submission.csv` output.
- [x] **Schedule Alignment:** Timeline mapped across Weeks 1–12 with specific milestone checkpoints.
- [x] **Review & Commit:** Approved by Team Lead and committed to `docs/test_plan.md`.