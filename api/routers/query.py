
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List


# Request Models
# ==========================
class KISRequest(BaseModel):
    query_id: str | None = None
    query: str
    top_k: int = 5


class TrakeRequest(BaseModel):
    query_id: str | None = None
    events: List[str]
    top_k: int = 3


class QARequest(BaseModel):
    query_id: str | None = None
    scene: str
    question: str
    top_k: int = 3


class KISBatchRequest(BaseModel):
    queries: List[KISRequest]


class TrakeBatchRequest(BaseModel):
    queries: List[TrakeRequest]


class QABatchRequest(BaseModel):
    queries: List[QARequest]

router = APIRouter()

@router.get("/")
def home():
    return {
        "message": "Mock API Server is running!"
    }


@router.post("/query/kis")
def query_kis(request: KISRequest):
    print(request.query)
    print(request.top_k)
     # Validation
    if request.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0"
        )

    if request.query.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="query cannot be empty"
        )

    return {
        "results": [
            {
                "video_id": "L01_V001",
                "video_fps": 25,
                "frame_index": 235,
                "start_frame": 220,
                "end_frame": 250,
                "timestamp_seconds": 9.40,
                "score": 0.98
            },
            {
                "video_id": "L01_V008",
                "video_fps": 25,
                "frame_index": 412,
                "start_frame": 400,
                "end_frame": 430,
                "timestamp_seconds": 16.48,
                "score": 0.95
            },
            {
                "video_id": "L02_V001",
                "video_fps": 25,
                "frame_index": 112,
                "start_frame": 100,
                "end_frame": 125,
                "timestamp_seconds": 4.48,
                "score": 0.93
            },
            {
                "video_id": "L03_V010",
                "video_fps": 30,
                "frame_index": 889,
                "start_frame": 870,
                "end_frame": 910,
                "timestamp_seconds": 29.63,
                "score": 0.91
            },
            {
                "video_id": "L05_V003",
                "video_fps": 25,
                "frame_index": 420,
                "start_frame": 400,
                "end_frame": 440,
                "timestamp_seconds": 16.80,
                "score": 0.88
            }
        ],
        "total_results": 5,
        "has_next": False,
        "query_time_ms": 340
    }


@router.post("/query/trake")
def query_trake(request: TrakeRequest):
    print(request.events)
    # Validation
    if len(request.events) == 0:
        raise HTTPException(
            status_code=400,
            detail="events cannot be empty"
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0"
        )


    return {
    "results": [
        {
            "video_id": "L01_V001",
            "video_fps": 25,
            "overall_score": 0.87,
            "events": [
                {
                    "event_index": 0,
                    "description": "Người đàn ông mặc áo đỏ bước lên sân khấu",
                    "frame_index": 512,
                    "start_frame": 490,
                    "end_frame": 540,
                    "timestamp_seconds": 20.48,
                    "score": 0.91,
                    "found": True
                },
                {
                    "event_index": 1,
                    "description": "Cô gái cầm micro bắt đầu hát",
                    "frame_index": 870,
                    "start_frame": 850,
                    "end_frame": 910,
                    "timestamp_seconds": 34.80,
                    "score": 0.84,
                    "found": True
                },
                {
                    "event_index": 2,
                    "description": "Khán giả đứng dậy vỗ tay",
                    "frame_index": 1203,
                    "start_frame": 1190,
                    "end_frame": 1240,
                    "timestamp_seconds": 48.12,
                    "score": 0.31,
                    "found": False
                }
            ]
        },
        {
            "video_id": "L02_V003",
            "video_fps": 25,
            "overall_score": 0.81,
            "events": [
                {
                    "event_index": 0,
                    "description": "Người đàn ông mặc áo đỏ bước lên sân khấu",
                    "frame_index": 300,
                    "start_frame": 280,
                    "end_frame": 320,
                    "timestamp_seconds": 12.00,
                    "score": 0.83,
                    "found": True
                },
                {
                    "event_index": 1,
                    "description": "Cô gái cầm micro bắt đầu hát",
                    "frame_index": 650,
                    "start_frame": 630,
                    "end_frame": 680,
                    "timestamp_seconds": 26.00,
                    "score": 0.79,
                    "found": True
                },
                {
                    "event_index": 2,
                    "description": "Khán giả đứng dậy vỗ tay",
                    "frame_index": 900,
                    "start_frame": 880,
                    "end_frame": 930,
                    "timestamp_seconds": 36.00,
                    "score": 0.42,
                    "found": False
                }
            ]
        },
        {
            "video_id": "L03_V005",
            "video_fps": 30,
            "overall_score": 0.76,
            "events": [
                {
                    "event_index": 0,
                    "description": "Người đàn ông mặc áo đỏ bước lên sân khấu",
                    "frame_index": 210,
                    "start_frame": 200,
                    "end_frame": 230,
                    "timestamp_seconds": 7.00,
                    "score": 0.80,
                    "found": True
                },
                {
                    "event_index": 1,
                    "description": "Cô gái cầm micro bắt đầu hát",
                    "frame_index": 520,
                    "start_frame": 500,
                    "end_frame": 540,
                    "timestamp_seconds": 17.33,
                    "score": 0.72,
                    "found": True
                },
                {
                    "event_index": 2,
                    "description": "Khán giả đứng dậy vỗ tay",
                    "frame_index": 830,
                    "start_frame": 810,
                    "end_frame": 850,
                    "timestamp_seconds": 27.67,
                    "score": 0.30,
                    "found": False
                }
            ]
        }
    ],
    "total_results": 3,
    "has_next": False,
    "query_time_ms": 610
}


@router.post("/query/qa")
def query_qa(request: QARequest):
    print(request.scene)
    print(request.question)

    # Validation
    if request.question.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="question cannot be empty"
        )

    if request.scene.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="scene cannot be empty"
        )

    if request.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0"
        )
    
    return {
    "results": [
    {
        "video_id": "L01_V001",
        "video_fps": 25,
        "frame_index": 2310,
        "start_frame": 2290,
        "end_frame": 2350,
        "timestamp_seconds": 92.4,
        "answer": "Nguyễn Văn An",
        "answer_source": "ocr",
        "score": 0.79
    },
    {
        "video_id": "L02_V004",
        "video_fps": 30,
        "frame_index": 870,
        "start_frame": 850,
        "end_frame": 910,
        "timestamp_seconds": 29.0,
        "answer": "Trần Minh Khoa",
        "answer_source": "llm",
        "score": 0.61
    },
    {
        "video_id": "L03_V002",
        "video_fps": 25,
        "frame_index": 1540,
        "start_frame": 1520,
        "end_frame": 1560,
        "timestamp_seconds": 61.6,
        "answer": "Lê Thị Hương",
        "answer_source": "ocr",
        "score": 0.57
    }
],
"total_results": 3,
"has_next": False,
"query_time_ms": 820
    }

@router.post("/query/kis/batch")
def query_kis_batch(request: KISBatchRequest):
  print(request.queries)
  
  return {
        "results": [
            {
                "query_id": "p1-1",
                "results": [
                    {
                        "video_id": "L01_V001",
                        "video_fps": 25,
                        "frame_index": 235,
                        "start_frame": 220,
                        "end_frame": 250,
                        "timestamp_seconds": 9.4,
                        "score": 0.98
                    }
                ],
                "total_results": 1,
                "has_next": False
            },

            {
                "query_id": "p1-2",
                "results": [
                    {
                        "video_id": "L02_V003",
                        "video_fps": 25,
                        "frame_index": 530,
                        "start_frame": 510,
                        "end_frame": 560,
                        "timestamp_seconds": 21.2,
                        "score": 0.81
                    }
                ],
                "total_results": 1,
                "has_next": False
            }
        ],
        "total_query_time_ms": 1240
    }


@router.post("/query/trake/batch")
def query_trake_batch(request: TrakeBatchRequest):
  print(request.queries)
  return {
        "results": [
            {
                "query_id": "p1-5",
                "results": [
                    {
                        "video_id": "L01_V001",
                        "video_fps": 25,
                        "overall_score": 0.87,
                        "events": [
                            {
                                "event_index": 0,
                                "description": "Người đàn ông mặc áo đỏ bước lên sân khấu",
                                "frame_index": 512,
                                "start_frame": 490,
                                "end_frame": 540,
                                "timestamp_seconds": 20.48,
                                "score": 0.91,
                                "found": True
                            },
                            {
                                "event_index": 1,
                                "description": "Cô gái cầm micro bắt đầu hát",
                                "frame_index": 870,
                                "start_frame": 850,
                                "end_frame": 910,
                                "timestamp_seconds": 34.80,
                                "score": 0.84,
                                "found": True
                            },
                            {
                                "event_index": 2,
                                "description": "Khán giả đứng dậy vỗ tay",
                                "frame_index": 1203,
                                "start_frame": 1190,
                                "end_frame": 1240,
                                "timestamp_seconds": 48.12,
                                "score": 0.31,
                                "found": False
                            }
                        ]
                    }
                ],
                "total_results": 1,
                "has_next": False
            },

            {
                "query_id": "p1-6",
                "results": [
                    {
                        "video_id": "L02_V003",
                        "video_fps": 25,
                        "overall_score": 0.82,
                        "events": [
                            {
                                "event_index": 0,
                                "description": "Người đàn ông mặc áo đỏ bước lên sân khấu",
                                "frame_index": 320,
                                "start_frame": 300,
                                "end_frame": 340,
                                "timestamp_seconds": 12.80,
                                "score": 0.86,
                                "found": True
                            },
                            {
                                "event_index": 1,
                                "description": "Cô gái cầm micro bắt đầu hát",
                                "frame_index": 680,
                                "start_frame": 660,
                                "end_frame": 700,
                                "timestamp_seconds": 27.20,
                                "score": 0.77,
                                "found": True
                            },
                            {
                                "event_index": 2,
                                "description": "Khán giả đứng dậy vỗ tay",
                                "frame_index": 950,
                                "start_frame": 930,
                                "end_frame": 970,
                                "timestamp_seconds": 38.00,
                                "score": 0.42,
                                "found": False
                            }
                        ]
                    }
                ],
                "total_results": 1,
                "has_next": False
            }
        ],
        "total_query_time_ms": 1850
    }


@router.post("/query/qa/batch")
def query_qa_batch(request: QABatchRequest):
  print(request.queries)
  return {
        "results": [
            {
                "query_id": "p1-3",
                "results": [
                    {
                        "video_id": "L01_V001",
                        "video_fps": 25,
                        "frame_index": 2310,
                        "start_frame": 2290,
                        "end_frame": 2350,
                        "timestamp_seconds": 92.4,
                        "answer": "Nguyễn Văn An",
                        "answer_source": "ocr",
                        "score": 0.79
                    }
                ],
                "total_results": 1,
                "has_next": False
            },

            {
                "query_id": "p1-4",
                "results": [
                    {
                        "video_id": "L02_V004",
                        "video_fps": 30,
                        "frame_index": 870,
                        "start_frame": 850,
                        "end_frame": 910,
                        "timestamp_seconds": 29.0,
                        "answer": "Trần Minh Khoa",
                        "answer_source": "llm",
                        "score": 0.61
                    }
                ],
                "total_results": 1,
                "has_next": False
            },

            {
                "query_id": "p1-5",
                "results": [
                    {
                        "video_id": "L03_V002",
                        "video_fps": 25,
                        "frame_index": 1540,
                        "start_frame": 1520,
                        "end_frame": 1560,
                        "timestamp_seconds": 61.6,
                        "answer": "Lê Thị Hương",
                        "answer_source": "ocr",
                        "score": 0.57
                    }
                ],
                "total_results": 1,
                "has_next": False
            }
        ],
        "total_query_time_ms": 1640
    }
