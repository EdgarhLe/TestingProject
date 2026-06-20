"""
VL-JEPA API — FastAPI entry point
Chạy: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI

app = FastAPI(
    title="VL-JEPA Multimedia Retrieval API",
    description="KIS · Trake · Q&A — Vietnamese AI Challenge",
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# TODO Tuần 2: thêm router cho /query
# from api.routers import query
# app.include_router(query.router)
