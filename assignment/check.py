"""환경 점검 — 과제를 시작하기 전에 한 번 돌려 볼 것.

  python check.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ok = True


def line(name, good, note=""):
    global ok
    ok = ok and good
    print(f"{name:<22} {'OK' if good else 'FAIL':<6} {note}")


print()
v = sys.version_info
line("Python", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}")

try:
    import numpy as np
    line("numpy", True, np.__version__)
except Exception as e:
    line("numpy", False, str(e)[:60])

try:
    import sentence_transformers as st
    line("sentence-transformers", True, st.__version__)
except Exception as e:
    line("sentence-transformers", False, str(e)[:60])

docs = sorted((BASE / "docs").glob("*.md"))
line("문서", len(docs) > 0, f"{len(docs)}건")

try:
    import json
    qs = json.loads((BASE / "questions.json").read_text(encoding="utf-8"))
    line("질문", len(qs) > 0, f"{len(qs)}개")
except Exception as e:
    line("질문", False, str(e)[:60])

if ok:
    try:
        import harness
        v = harness.embed(["연결 확인용 문장"])
        line("임베딩 모델", v.shape[1] > 0, f"{v.shape[1]}차원")
    except Exception as e:
        line("임베딩 모델", False, str(e)[:80])

print()
print("→ 준비 완료" if ok else "→ FAIL 항목을 먼저 해결할 것")
