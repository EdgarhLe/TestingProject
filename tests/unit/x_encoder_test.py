import pytest


def test_vjepa2_encoder_features() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    video_decoder_module = pytest.importorskip("torchcodec.decoders")
    AutoModel = transformers.AutoModel
    AutoVideoProcessor = transformers.AutoVideoProcessor
    VideoDecoder = video_decoder_module.VideoDecoder

    hf_repo = "facebook/vjepa2-vitl-fpc64-256"
    video_url = (
        "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/"
        "archery/-Qz25rXdMjE_000014_000024.mp4"
    )

    model = AutoModel.from_pretrained(hf_repo)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    vr = VideoDecoder(video_url)

    frame_idx = np.arange(0, 64)
    video = vr.get_frames_at(indices=frame_idx).data
    inputs = processor(video, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        video_embeddings = model.get_vision_features(**inputs)

    assert video_embeddings is not None
