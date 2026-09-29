import json
import sys
import requests
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BENCHMARK_TEST_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_grounded": True,
        "description": "Definition of Agentic AI"
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_grounded": True,
        "description": "AI agents vs traditional automation"
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_grounded": True,
        "description": "Core architecture components"
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_grounded": True,
        "description": "Role of memory in agent workflows"
    },
    {
        "id": 5,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_grounded": False,
        "description": "Out-of-domain query (refusal / hallucination prevention test)"
    },
    {
        "id": 6,
        "query": "What are common design patterns for multi-agent collaboration?",
        "expected_grounded": True,
        "description": "Multi-agent system design patterns"
    }
]


def run_via_api(url: str, query: str) -> Dict[str, Any]:
    """Execute query via FastAPI endpoint."""
    response = requests.post(f"{url}/chat", json={"query": query}, timeout=60)
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data["final_answer"],
        "context": data["retrieved_context"],
        "score": data["confidence_score"]
    }


_GRAPH_INSTANCE = None


def run_via_direct_graph(query: str) -> Dict[str, Any]:
    """Execute query via direct LangGraph invocation."""
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is None:
        from src.graph import build_rag_graph
        _GRAPH_INSTANCE = build_rag_graph()

    initial_state = {
        "question": query,
        "context": [],
        "answer": "",
        "score": 0.0,
        "sources": []
    }
    result = _GRAPH_INSTANCE.invoke(initial_state)
    return {
        "answer": result.get("answer", ""),
        "context": result.get("context", []),
        "score": result.get("score", 0.0)
    }


def check_fastapi_alive(url: str = "http://localhost:8000") -> bool:
    """Check if FastAPI server is responsive."""
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def run_all_tests():
    print("=" * 80)
    print("      AGENTIC AI RAG SYSTEM - BENCHMARK & GROUNDING VERIFICATION SUITE")
    print("=" * 80)

    api_url = "http://localhost:8000"
    use_api = check_fastapi_alive(api_url)

    if use_api:
        print(f"[Test Runner] Connected to running FastAPI server at {api_url}")
    else:
        print("[Test Runner] FastAPI not detected locally. Using direct LangGraph engine.")

    results_summary = []

    for test in BENCHMARK_TEST_QUERIES:
        qid = test["id"]
        query = test["query"]
        expected_grounded = test["expected_grounded"]
        desc = test["description"]

        print(f"\n--- [Test #{qid}] {desc} ---")
        print(f"Query: \"{query}\"")

        try:
            if use_api:
                res = run_via_api(api_url, query)
            else:
                res = run_via_direct_graph(query)

            answer = res["answer"]
            context = res["context"]
            score = res["score"]

            print(f"Retrieved Chunks Count: {len(context)}")
            if context:
                snippet = context[0][:200].replace("\n", " ") + "..."
                print(f"Top Context Snippet: {snippet}")

            print(f"Generated Answer:\n{answer}")
            print(f"Confidence Score: {score}")

            # Verification logic
            is_refusal = (
                "cannot answer based on the provided document" in answer.lower()
                or score == 0.0
            )

            if expected_grounded:
                # Should NOT refuse and should have positive confidence
                passed = (not is_refusal) and (score > 0.0) and (len(answer.strip()) > 20)
                status_str = "PASS" if passed else "FAIL (Unexpected refusal or low confidence)"
            else:
                # Out-of-domain: MUST refuse and have zero confidence score
                passed = is_refusal or (score == 0.0)
                status_str = "PASS (Properly Refused)" if passed else "FAIL (Hallucination detected)"

            print(f"Result: {status_str}")
            results_summary.append({
                "id": qid,
                "query": query,
                "passed": passed,
                "status": status_str,
                "score": score
            })

        except Exception as e:
            print(f"Execution Error: {e}")
            results_summary.append({
                "id": qid,
                "query": query,
                "passed": False,
                "status": f"ERROR: {str(e)}",
                "score": 0.0
            })

    print("\n" + "=" * 80)
    print("                             TEST SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ID':<4} | {'Status':<32} | {'Score':<6} | {'Query'}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['id']:<4} | {r['status']:<32} | {r['score']:<6.2f} | {r['query'][:35]}...")
    print("=" * 80)

    total_passed = sum(1 for r in results_summary if r["passed"])
    print(f"Total Passed: {total_passed} / {len(results_summary)}")


if __name__ == "__main__":
    run_all_tests()
