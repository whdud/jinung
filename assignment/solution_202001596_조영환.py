"""학생이 채우는 파일

낼 때 파일 이름을 solution_학번_이름.py 로 바꿀 것
내기 전에 python grade.py 가 끝까지 돌아가는지 확인할 것

[전략 요약]  Recall@5:  베이스라인 0.65  →  최종 0.90
  ① chunk()   구조 기반 청킹 — "## 제N조" 단위로 자름              (0.65 → 0.85)
  ② search()  하이브리드 검색 — 벡터(의미) + BM25(키워드) 를 RRF 로 합침  (0.85 → 0.90)

외부 라이브러리는 numpy 와 harness 만 사용함 (채점 PC 에 추가 설치가 없어도 돌아가도록).
"""
import math
import re
from collections import Counter

import numpy as np

import harness

# ---------------------------------------------------------------- #
#  ① chunk  —  문서를 조각으로 자르기
# ---------------------------------------------------------------- #


def chunk(text: str) -> list[str]:
    """문서 전문을 받아 조각 리스트로 자름.

    구조 기반 청킹 — 문서가 이미 "## 제N조 (제목)" 으로 나뉘어 있으므로
    "## " 로 시작하는 줄 앞에서 잘라 조(條) 하나를 조각 하나로 만듦.

    왜 이렇게?
      - 글자 수(500자)로 자르면 문장이 중간에서 끊기고 두 조항이 한 조각에 섞임.
      - 문장 하나씩 자르면 "이 경우…" 처럼 주제어가 없는 조각이 생겨 오히려 점수가 떨어짐(0.50).
      - 조 하나 = 한 가지 주제 + 제목 포함 → 질문과 맞는 단위.
    """
    return re.split(r"\n(?=## )", text)


# ---------------------------------------------------------------- #
#  ② search  —  질문에 가까운 조각 찾기
# ---------------------------------------------------------------- #
#
#  두 가지 검색기를 함께 씀 (하이브리드 검색)
#    - 벡터(밀집) 검색 : 뜻이 비슷한 조각에 강함  (동의어: "그만두려면" ↔ "자퇴")
#    - BM25(희소) 검색 : 글자가 똑같은 조각에 강함 (코드·고유명사: "KCC-5", "SCH-02")
#  두 검색기의 점수는 단위가 달라 더할 수 없으므로 "순위"만 뽑아 RRF 로 합침.

# ---- 1) 토큰화: 글을 BM25 가 셀 수 있는 낱말 조각으로 쪼갬 -------- #

def _tokenize(text: str) -> list[str]:
    """영문/코드는 통째로, 한글은 두 글자씩(bigram) 쪼갬.

    한글은 조사가 붙어("학기를", "학기는") 띄어쓰기 단위로는 같은 단어가 서로 다르게 보임.
    두 글자씩 쪼개면 "학기"가 공통으로 잡힘.   "수강인원" -> "수강", "강인", "인원"
    영문·숫자 코드는 "kcc-5", "sch-02" 처럼 하나의 토큰으로 유지해서 정확히 일치시킴.
    """
    tokens: list[str] = []
    for m in re.finditer(r"[a-z]+(?:-\d+)?|\d+|[가-힣]+", text.lower()):
        w = m.group()
        if "가" <= w[0] <= "힣":
            tokens += [w[i:i + 2] for i in range(len(w) - 1)] if len(w) > 1 else [w]
        else:
            tokens.append(w)
    return tokens


# ---- 2) BM25 색인: 조각이 바뀔 때만 한 번 만들어 둠 ----------------- #

_K1 = 1.5    # 단어가 여러 번 나와도 점수가 무한히 늘지 않게 하는 정도 (보통 1.2~2.0)
_B = 0.75    # 긴 조각에 불이익을 주는 정도 (0 = 길이 무시, 1 = 완전 반영)
_RRF_K = 60  # RRF 상수 (표준값). 클수록 1·2위 차이를 완만하게 만듦

_bm25 = {"key": None}


def _build_bm25() -> None:
    chunks = harness.CHUNKS
    key = (len(chunks), id(chunks), hash(tuple(chunks)))
    if _bm25["key"] == key:
        return
    tf = [Counter(_tokenize(c)) for c in chunks]            # 조각별 단어 빈도(TF)
    df: Counter = Counter()                                  # 단어가 등장한 조각 수
    for c in tf:
        df.update(c.keys())
    n = len(chunks)
    lengths = np.array([sum(c.values()) for c in tf], dtype=float)
    _bm25.update(
        key=key,
        tf=tf,
        lengths=lengths,
        avg_len=lengths.mean(),
        # IDF: 드문 단어일수록 큰 값 (모든 조각에 있는 단어는 0 에 가까움)
        idf={t: math.log((n - d + 0.5) / (d + 0.5) + 1) for t, d in df.items()},
    )


def _bm25_scores(question: str) -> np.ndarray:
    """질문의 단어들이 각 조각에 얼마나 잘 맞는지 BM25 점수를 계산 -> (N,)"""
    _build_bm25()
    tf, lengths, avg_len, idf = (_bm25[x] for x in ("tf", "lengths", "avg_len", "idf"))
    scores = np.zeros(len(tf))
    for term in set(_tokenize(question)):
        if term not in idf:
            continue                                         # 어떤 조각에도 없는 단어는 무시
        for i, counts in enumerate(tf):
            f = counts.get(term, 0)                          # 이 조각에서 term 이 나온 횟수
            if f:
                norm_len = 1 - _B + _B * lengths[i] / avg_len
                scores[i] += idf[term] * f * (_K1 + 1) / (f + _K1 * norm_len)
    return scores


# ---- 3) RRF: 여러 검색기의 순위를 하나로 합침 --------------------- #

def _ranks(scores: np.ndarray) -> np.ndarray:
    """점수 -> 순위 (가장 높은 점수가 1위)"""
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _rrf(*score_lists: np.ndarray) -> np.ndarray:
    """Reciprocal Rank Fusion:  최종점수 = Σ 1 / (60 + 순위)

    두 검색기에서 모두 상위권인 조각이 가장 높은 점수를 받음.
    점수 크기는 버리고 순위만 쓰기 때문에 코사인(0~1)과 BM25(무한대)를 그대로 합칠 수 있음.
    """
    return sum(1.0 / (_RRF_K + _ranks(s)) for s in score_lists)


# ---- 4) search ---------------------------------------------------- #

def search(question: str, k: int) -> list[str]:
    """질문을 받아 상위 k개 조각의 원문을 돌려줌."""
    qv = harness.embed([question])[0]        # (D,)  질문 벡터
    dense = harness.VECS @ qv                # (N,)  벡터 검색: 의미가 비슷한 정도(코사인 유사도)
    sparse = _bm25_scores(question)          # (N,)  BM25 검색: 글자가 일치하는 정도
    fused = _rrf(dense, sparse)              # (N,)  두 순위를 합친 최종 점수
    top = np.argsort(-fused)[:k]             # 큰 순서로 k개
    return [harness.CHUNKS[i] for i in top]
