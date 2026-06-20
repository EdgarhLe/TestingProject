# VL-JEPA — Hệ Thống Tìm Kiếm Đa Phương Tiện

Vietnamese AI Challenge — Video Retrieval | KIS • Trake • Q&A  
Nhóm 5 người • 12 tuần • 2 GPU (Máy 1: training, Máy 2: dev/inference)

---

## Cấu Trúc Thư Mục

```
vl-jepa/
│
├── model/                  # Kiến trúc mô hình VL-JEPA
│   ├── x_encoder/          # V-JEPA 2 ViT-L (đóng băng) — bộ mã hóa thị giác
│   ├── predictor/          # Qwen2.5 upper layers — dự đoán embedding
│   ├── y_encoder/          # bge-m3 — mã hóa văn bản/truy vấn
│   └── y_decoder/          # Decoder nhẹ — chỉ dùng lúc inference (Q&A)
│
├── training/               # Huấn luyện mô hình (chạy trên Máy 1)
│   ├── data/               # DataLoader, tiền xử lý, data pipeline
│   ├── losses/             # Bidirectional InfoNCE loss
│   └── checkpoints/        # Checkpoint huấn luyện (KHÔNG commit lên git)
│
├── indexing/               # Pipeline đánh chỉ mục video (chạy trên Máy 2)
│   ├── segmentation/       # Phân đoạn ngữ nghĩa (sliding window + clustering)
│   ├── indexes/            # FAISS HNSW (coarse + fine) và BM25
│   ├── ocr/                # PaddleOCR — trích xuất văn bản trên màn hình
│   └── asr/                # Whisper — transcript với timestamp
│
├── api/                    # FastAPI backend — phục vụ truy vấn
│   ├── routers/            # Endpoint cho KIS, Trake, Q&A
│   ├── schemas/            # Pydantic request/response schemas
│   └── services/           # Logic truy vấn, kết nối index
│
├── frontend/               # Giao diện web
│   └── src/
│       ├── components/
│       │   ├── search/     # Ô nhập truy vấn, chọn loại nhiệm vụ
│       │   ├── results/    # Hiển thị kết quả KIS / Trake / Q&A
│       │   └── player/     # Video player với tua đến frame
│       ├── pages/          # Trang chính
│       └── api/            # API client (gọi backend)
│
├── tests/
│   ├── unit/               # Test từng module riêng lẻ
│   ├── integration/        # Test luồng kết hợp nhiều module
│   └── e2e/                # Test end-to-end toàn hệ thống
│
├── scripts/                # Script tiện ích (rsync checkpoint, chạy indexing, v.v.)
├── docs/                   # Tài liệu kỹ thuật bổ sung
└── configs/                # File cấu hình (model, training, indexing, API)
```

---

## Quy Tắc Commit

```
<loại>(<phạm vi>): <mô tả ngắn>

Ví dụ:
feat(model): thêm bidirectional attention mask cho Predictor
fix(api): sửa lỗi schema response Trake thiếu trường frame_index
chore(training): cập nhật config curriculum Stage B
test(indexing): thêm unit test cho segmentation pipeline
docs(api): cập nhật API contract v0
```

**Loại commit:** `feat` · `fix` · `chore` · `test` · `docs` · `refactor`  
**Phạm vi:** `model` · `training` · `indexing` · `api` · `frontend` · `tests` · `scripts`

Commit thẳng vào `main` cho tuần đầu. Từ Tuần 2 trở đi dùng feature branch + PR.

---

## Phân Công Thư Mục Theo Người

| Thư mục | Người phụ trách chính |
|---|---|
| `model/` | L (Leader) |
| `training/` | L (Leader) |
| `indexing/` | DE (Data Engineer) |
| `api/` | L (Leader) |
| `frontend/` | FE1, FE2 |
| `tests/` | QA (cùng với cả nhóm) |
| `scripts/` | L, DE |
| `configs/` | L |
| `docs/` | Cả nhóm |

---

## Thiết Lập Môi Trường

### Backend (Python)

```bash
# Clone repo
git clone <repo-url>
cd vl-jepa

# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Biến Môi Trường

Copy file `.env.example` thành `.env` và điền các giá trị thực:

```bash
cp .env.example .env
```

---

## Quy Ước Đường Dẫn Dữ Liệu

Để code không bị hardcode đường dẫn khác nhau giữa máy, dùng biến môi trường trong `.env`:

```
DATA_ROOT=/path/to/ai-challenge-dataset   # thư mục gốc chứa video
INDEX_ROOT=/path/to/indexes               # thư mục chứa FAISS + BM25 indexes
CHECKPOINT_DIR=/path/to/checkpoints       # thư mục checkpoint (Máy 1)
```

Trong code, luôn đọc từ biến môi trường, không hardcode path.

---

## Quy Tắc Quan Trọng

- **Máy 1 chỉ dùng để training (Tuần 3–8).** Mọi dev, test, inference đều chạy trên Máy 2.
- **Không commit checkpoint** vào git (đã có trong `.gitignore`). Checkpoint được đồng bộ qua rsync.
- **Không commit file `.env`** chứa thông tin nhạy cảm.

---

## Tài Liệu Liên Quan

- [`docs/api_contract_v0.md`](docs/api_contract_v0.md) — API contract v0 (KIS / Trake / Q&A)
- [`docs/system_design.md`](docs/system_design.md) — Thiết kế hệ thống đầy đủ
- [`configs/`](configs/) — Cấu hình model, training, indexing
