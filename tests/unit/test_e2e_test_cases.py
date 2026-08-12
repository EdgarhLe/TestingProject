import json
from pathlib import Path
import pytest

TEST_CASES_DIR = Path(__file__).parent.parent / "e2e" / "test_cases"

def test_kis_test_cases_schema_and_counts():
    file_path = TEST_CASES_DIR / "kis_test_cases.json"
    assert file_path.exists(), f"File {file_path} does not exist"
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) >= 6, "Must have at least 6 total KIS test cases"
    
    sample_count = sum(1 for item in data if item.get("source") == "sample")
    synthetic_count = sum(1 for item in data if item.get("source") == "synthetic")
    
    assert sample_count >= 2, "Must have at least 2 sample KIS test cases"
    assert synthetic_count >= 3, "Must have at least 3 synthetic KIS test cases"
    
    for item in data:
        assert "test_id" in item
        assert "query_id" in item
        assert "query_text" in item and len(item["query_text"]) > 0
        assert "expected" in item
        assert "difficulty" in item and item["difficulty"] in ("easy", "medium", "hard")
        assert "source" in item and item["source"] in ("sample", "synthetic")
        
        expected = item["expected"]
        assert "video_id" in expected and isinstance(expected["video_id"], str)
        assert "frame_index" in expected and isinstance(expected["frame_index"], int) and expected["frame_index"] >= 0

def test_qa_test_cases_schema_and_counts():
    file_path = TEST_CASES_DIR / "qa_test_cases.json"
    assert file_path.exists(), f"File {file_path} does not exist"
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) >= 6, "Must have at least 6 total Q&A test cases"
    
    sample_count = sum(1 for item in data if item.get("source") == "sample")
    synthetic_count = sum(1 for item in data if item.get("source") == "synthetic")
    
    assert sample_count >= 2, "Must have at least 2 sample Q&A test cases"
    assert synthetic_count >= 3, "Must have at least 3 synthetic Q&A test cases"
    
    for item in data:
        assert "test_id" in item
        assert "query_id" in item
        assert "query_text" in item and len(item["query_text"]) > 0
        assert "expected" in item
        assert "difficulty" in item and item["difficulty"] in ("easy", "medium", "hard")
        assert "source" in item and item["source"] in ("sample", "synthetic")
        
        expected = item["expected"]
        assert "video_id" in expected and isinstance(expected["video_id"], str)
        assert "frame_index" in expected and isinstance(expected["frame_index"], int) and expected["frame_index"] >= 0
        assert "answer" in expected and isinstance(expected["answer"], str) and len(expected["answer"]) > 0

def test_trake_test_cases_schema_and_counts():
    file_path = TEST_CASES_DIR / "trake_test_cases.json"
    assert file_path.exists(), f"File {file_path} does not exist"
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) >= 6, "Must have at least 6 total TRAKE test cases"
    
    sample_count = sum(1 for item in data if item.get("source") == "sample")
    synthetic_count = sum(1 for item in data if item.get("source") == "synthetic")
    
    assert sample_count >= 2, "Must have at least 2 sample TRAKE test cases"
    assert synthetic_count >= 3, "Must have at least 3 synthetic TRAKE test cases"
    
    for item in data:
        assert "test_id" in item
        assert "query_id" in item
        assert "query_text" in item and len(item["query_text"]) > 0
        assert "expected" in item
        assert "difficulty" in item and item["difficulty"] in ("easy", "medium", "hard")
        assert "source" in item and item["source"] in ("sample", "synthetic")
        
        expected = item["expected"]
        assert "video_id" in expected and isinstance(expected["video_id"], str)
        assert "events" in expected and isinstance(expected["events"], list) and len(expected["events"]) > 0
        for event in expected["events"]:
            assert "event_index" in event and isinstance(event["event_index"], int)
            assert "frame_index" in event and isinstance(event["frame_index"], int) and event["frame_index"] >= 0
