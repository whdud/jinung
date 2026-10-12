"""학생이 채우는 파일 

낼 때 파일 이름을 solution_학번_이름.py 로 바꿀 것
내기 전에 python grade.py 가 끝까지 돌아가는지 확인할 것
"""
import numpy as np

import harness

# ---------------------------------------------------------------- #
#  ① chunk  —  문서를 조각으로 자르기
# ---------------------------------------------------------------- #

SIZE = 500          # 조각 하나의 글자 수


def chunk(text: str) -> list[str]:
    """문서 전문을 받아 조각 리스트로 자름.

    베이스라인 — 글자 수로만 기계적으로 자름.
    문장도 단락도 제목도 보지 않고, 오버랩도 없음.
    """
    return [text[i:i + SIZE] for i in range(0, len(text), SIZE)]


# ---------------------------------------------------------------- #
#  ② search  —  질문에 가까운 조각 찾기
# ---------------------------------------------------------------- #

def search(question: str, k: int) -> list[str]:
    """질문을 받아 상위 k개 조각의 원문을 돌려줌.

    harness.VECS 는 정규화된 (N, D) 행렬이라
    행렬곱 한 번이면 모든 조각과의 코사인 유사도가 나옴.
    """
    qv = harness.embed([question])[0]        # (D,)  질문 벡터
    sims = harness.VECS @ qv                 # (N,)  조각마다 코사인 유사도 
    top = np.argsort(-sims)[:k]              # 큰 순서로 k개 
    return [harness.CHUNKS[i] for i in top]
