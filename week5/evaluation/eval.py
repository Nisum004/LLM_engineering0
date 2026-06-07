import sys
import math
import os
from pathlib import Path
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

# Ensure `evaluation` and `RAG` imports work when this file is run directly.
CURRENT_DIR = Path(__file__).resolve().parent
WEEK5_ROOT = CURRENT_DIR.parent
if str(WEEK5_ROOT) not in sys.path:
    sys.path.insert(0, str(WEEK5_ROOT))

from evaluation.test import TestQuestion, load_tests
from RAG.answer import answer_question, fetch_context

load_dotenv(override=True)

# =========================
# CONFIG
# =========================

MODEL = os.getenv("EVAL_MODEL", "ollama/llama3.1:8b")
# Direct Ollama (default). For LiteLLM proxy on :4000, set OPENAI_API_BASE=http://localhost:4000/v1
API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434")

# IMPORTANT: LiteLLM requires API key even for local servers
API_KEY = os.getenv("OPENAI_API_KEY", "anything")

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = API_BASE


def _judge_completion_kwargs() -> dict:
    """Build kwargs for litellm.completion (judge LLM)."""
    kwargs = {"model": MODEL, "api_base": API_BASE, "api_key": API_KEY}
    # LiteLLM proxy on :4000 speaks OpenAI API, not Ollama's /api/generate
    if ":4000" in API_BASE:
        kwargs["custom_llm_provider"] = "openai"
    return kwargs


db_name = "vector_db"


# =========================
# SCHEMAS
# =========================

class RetrievalEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain")
    keywords_found: int
    total_keywords: int
    keyword_coverage: float


class AnswerEval(BaseModel):
    feedback: str
    accuracy: float
    completeness: float
    relevance: float


# =========================
# RETRIEVAL METRICS
# =========================

def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()

    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    retrieved_docs = fetch_context(test.question)

    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]

    keywords_found = sum(1 for s in mrr_scores if s > 0)
    total_keywords = len(test.keywords)

    return RetrievalEval(
        mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        ndcg=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=(keywords_found / total_keywords * 100) if total_keywords else 0.0,
    )


# =========================
# ANSWER EVALUATION (LLM JUDGE)
# =========================

def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    generated_answer, retrieved_docs = answer_question(test.question)

    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator. Score strictly and fairly.",
        },
        {
            "role": "user",
            "content": f"""
Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Evaluate:
- Accuracy (1-5)
- Completeness (1-5)
- Relevance (1-5)
- Provide concise feedback
""",
        },
    ]

    judge_response = completion(
        messages=judge_messages,
        response_format=AnswerEval,
        max_retries=2,
        **_judge_completion_kwargs(),
    )

    answer_eval = AnswerEval.model_validate_json(
        judge_response.choices[0].message.content
    )

    return answer_eval, generated_answer, retrieved_docs


# =========================
# BATCH EVALUATION
# =========================

def evaluate_all_retrieval():
    tests = load_tests()
    total = len(tests)

    for i, test in enumerate(tests):
        yield test, evaluate_retrieval(test), (i + 1) / total


def evaluate_all_answers():
    tests = load_tests()
    total = len(tests)

    for i, test in enumerate(tests):
        yield test, evaluate_answer(test)[0], (i + 1) / total


# =========================
# CLI
# =========================

def run_cli_evaluation(test_number: int):
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print("Invalid test index")
        sys.exit(1)

    test = tests[test_number]

    print("\n" + "=" * 80)
    print(f"Test #{test_number}")
    print("=" * 80)
    print("Question:", test.question)
    print("Keywords:", test.keywords)
    print("Reference:", test.reference_answer)

    print("\nRetrieval Evaluation")
    retrieval = evaluate_retrieval(test)
    print(retrieval)

    print("\nAnswer Evaluation")
    answer_eval, gen_answer, _ = evaluate_answer(test)

    print("\nGenerated Answer:\n", gen_answer)
    print("\nFeedback:", answer_eval.feedback)
    print("Accuracy:", answer_eval.accuracy)
    print("Completeness:", answer_eval.completeness)
    print("Relevance:", answer_eval.relevance)


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval.py <test_index>")
        sys.exit(1)

    run_cli_evaluation(int(sys.argv[1]))


if __name__ == "__main__":
    main()