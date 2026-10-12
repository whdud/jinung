"""제공 코드 — 수정하지 말 것.

문서를 읽고, solution.chunk() 로 자르고, 임베딩해서 메모리에 올려 둠.
벡터 DB 는 쓰지 않음. 조각이 수백 개 수준이라 전수 비교가 더 빠름.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_NAME = "jhgan/ko-sroberta-multitask"   # 768차원 · 한국어 · CPU 로 동작
BASE = Path(__file__).resolve().parent
DOCS_DIR = BASE / "docs"
CACHE_DIR = BASE / ".cache"

# build() 가 채우는 전역 상태 — solution.search() 에서 읽어 씀
CHUNKS: list[str] = []          # 조각 원문
META: list[dict] = []           # 조각별 출처 {doc_id, chunk_idx}
VECS: np.ndarray | None = None  # (N, D) 단위벡터 행렬

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"임베딩 모델 불러오는 중 … {MODEL_NAME}")
        print("(처음 실행하면 모델을 내려받느라 몇 분 걸릴 수 있음)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """문장 리스트 -> (N, D) 실수 행렬.

    normalize_embeddings=True 라 모든 벡터의 길이가 1 임.
    따라서 내적이 곧 코사인 유사도가 됨.
    """
    model = _get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=len(texts) > 200,
    )
    return np.asarray(vecs, dtype=np.float32)


def load_docs() -> list[tuple[str, str]]:
    """docs/ 안의 .md 파일을 (문서이름, 전문) 목록으로 돌려줌."""
    files = sorted(DOCS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"문서를 찾을 수 없음: {DOCS_DIR}")
    return [(p.stem, p.read_text(encoding="utf-8")) for p in files]


MAX_CHUNKS = 5000   # 이보다 많으면 chunk() 가 잘못된 것으로 봄


def _as_pieces(ret, doc_id: str) -> list[str]:
    """chunk() 반환값을 검사해서 조각 리스트로 돌려줌. 틀리면 이유를 알려 줌."""
    if ret is None:
        raise TypeError(
            f"[{doc_id}] chunk() 가 아무것도 돌려주지 않음 — return 문이 있는지 확인할 것")
    if isinstance(ret, str):
        raise TypeError(
            f"[{doc_id}] chunk() 가 문자열 하나를 돌려줌 — 조각 '리스트'를 돌려줘야 함\n"
            "        return text[i:i + SIZE]\n"
            "        return [text[i:i + SIZE] for i in range(0, len(text), SIZE)]  ← 이렇게")
    try:
        pieces = list(ret)
    except TypeError:
        raise TypeError(
            f"[{doc_id}] chunk() 의 반환값을 조각 목록으로 볼 수 없음: {type(ret).__name__}") from None
    for i, p in enumerate(pieces):
        if not isinstance(p, str):
            raise TypeError(
                f"[{doc_id}] chunk() 가 돌려준 {i}번째 조각이 문자열이 아님: "
                f"{type(p).__name__}. 조각은 잘라낸 원문 문자열이어야 함")
    return pieces


def build(chunk_fn, use_cache: bool = True) -> int:
    """문서를 조각으로 자르고 임베딩해서 전역 상태에 올림. 조각 수를 돌려줌."""
    global CHUNKS, META, VECS

    CHUNKS, META = [], []
    for doc_id, text in load_docs():
        for i, piece in enumerate(_as_pieces(chunk_fn(text), doc_id)):
            piece = piece.strip()
            if piece:
                CHUNKS.append(piece)
                META.append({"doc_id": doc_id, "chunk_idx": i})

    if not CHUNKS:
        raise ValueError("조각이 하나도 만들어지지 않음 — chunk() 를 확인할 것")

    if len(CHUNKS) > MAX_CHUNKS:
        avg = sum(len(c) for c in CHUNKS) / len(CHUNKS)
        raise ValueError(
            f"조각이 {len(CHUNKS)}개나 만들어짐 (평균 {avg:.1f}자). "
            f"{MAX_CHUNKS}개가 넘으면 임베딩이 너무 오래 걸려 중단함.\n"
            "        조각 크기나 오버랩 보폭을 확인할 것")

    # 조각 내용이 같으면 임베딩을 다시 하지 않음
    key = hashlib.sha1(("\x00".join(CHUNKS) + MODEL_NAME).encode("utf-8")).hexdigest()[:16]
    cache = CACHE_DIR / f"{key}.npy"

    if use_cache and cache.exists():
        VECS = np.load(cache)
        print(f"조각 {len(CHUNKS)}개 — 캐시된 벡터를 사용함")
    else:
        print(f"조각 {len(CHUNKS)}개 — 임베딩 중 …")
        VECS = embed(CHUNKS)
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache, VECS)
        print(f"조각 {len(CHUNKS)}개 — 임베딩 완료  (shape={VECS.shape})")

    return len(CHUNKS)
