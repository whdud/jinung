"""제공 코드 — 수정하지 말 것.

사용법
  python grade.py            전체 채점
  python grade.py --show 7   7번 질문이 무엇을 가져왔는지 확인
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import harness
import solution

K = 5
BASE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    """공백을 전부 지워 비교 — 파싱·청킹에 따라 공백이 달라지기 때문."""
    return re.sub(r"\s+", "", s)


def dwidth(s: str) -> int:
    """한글은 두 칸을 차지하므로 표 정렬용 폭을 따로 계산."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - dwidth(s))


def load_questions() -> list[dict]:
    return json.loads((BASE / "questions.json").read_text(encoding="utf-8"))


class SearchError(Exception):
    """search() 의 반환값이 규칙에 맞지 않음."""


def ask(question: str, k: int, known: set[str], warned: list) -> list[str]:
    """solution.search() 를 호출하고 반환값이 규칙에 맞는지 확인함.

    규칙
      1. 조각 '원문 문자열' 의 목록일 것
      2. 개수가 k 를 넘지 않을 것
      3. 돌려준 조각은 실제로 만들어진 조각 중 하나일 것
    """
    got = solution.search(question, k)

    if got is None:
        raise SearchError("search() 가 아무것도 돌려주지 않음 — return 문이 있는지 확인할 것")
    if isinstance(got, str):
        raise SearchError("search() 가 문자열 하나를 돌려줌 — 조각 '리스트' 를 돌려줘야 함")
    try:
        got = list(got)
    except TypeError:
        raise SearchError(f"search() 의 반환값이 목록이 아님: {type(got).__name__}") from None

    for i, c in enumerate(got):
        if not isinstance(c, str):
            raise SearchError(
                f"search() 가 돌려준 {i}번째 항목이 문자열이 아님: {type(c).__name__}. "
                "조각 번호가 아니라 조각 원문을 돌려줘야 함")

    if len(got) > k:
        raise SearchError(
            f"search() 가 {len(got)}개를 돌려줌 — 최대 {k}개까지만 돌려줄 수 있음. "
            "[:k] 로 잘랐는지 확인할 것")

    outside = [c for c in got if c not in known]
    if outside and not warned:
        warned.append(True)
        print("경고 — search() 가 chunk() 로 만들어지지 않은 문자열을 돌려줌")
        print("       조각을 가공해서 돌려주고 있다면 harness.CHUNKS[i] 를 그대로 돌려줄 것")
        print(f"       예: {outside[0][:60]!r}")
        print()

    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, help="해당 번호 질문의 검색 결과를 출력")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    n_chunks = harness.build(solution.chunk)
    questions = load_questions()
    known = set(harness.CHUNKS)
    warned: list = []
    print()

    if args.show is not None:
        item = next((q for q in questions if q["id"] == args.show), None)
        if item is None:
            ids = ", ".join(str(q["id"]) for q in questions)
            print(f"{args.show}번 질문이 없음. 가능한 번호: {ids}")
            return
        got = ask(item["q"], args.k, known, warned)
        print(f"질문      {item['q']}")
        print(f"정답 구절  {item['answer_span']}")
        print(f"유형      {item['type']}")
        print("-" * 70)
        for i, c in enumerate(got, 1):
            mark = "O" if norm(item["answer_span"]) in norm(c) else " "
            body = c.replace("\n", " ")
            print(f"[{mark}] {i}위  {body[:160]}{'…' if len(body) > 160 else ''}")
            print()
        return

    rows = []
    for item in questions:
        got = ask(item["q"], args.k, known, warned)
        ok = any(norm(item["answer_span"]) in norm(c) for c in got)
        rows.append((item["id"], item["q"], item["type"], ok))

    w = max(dwidth(r[1]) for r in rows)
    for i, q, t, ok in rows:
        print(f"{i:2d}  {pad(q, w)}   {pad(t, 10)}  {'O' if ok else '-'}")

    hit = sum(ok for *_, ok in rows)
    print()
    print(f"조각 수       {n_chunks}")
    print(f"Recall@{args.k}     {hit} / {len(rows)}  =  {hit / len(rows):.3f}")
    print()

    miss = [(i, t) for i, q, t, ok in rows if not ok]
    if miss:
        by_type: dict[str, int] = {}
        for _, t in miss:
            by_type[t] = by_type.get(t, 0) + 1
        print("틀린 문항 유형 :", ", ".join(f"{k} {v}개" for k, v in by_type.items()))
        print("자세히 보려면  : python grade.py --show", miss[0][0])


if __name__ == "__main__":
    try:
        main()
    except (SearchError, TypeError, ValueError) as e:
        print()
        print("중단됨 —", e)
        print()
        raise SystemExit(1)
