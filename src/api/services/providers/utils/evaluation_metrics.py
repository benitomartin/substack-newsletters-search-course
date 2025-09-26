from opik.evaluation import models
from opik.evaluation.metrics import GEval

from src.config import settings
from src.utils.logger_util import setup_logging

logger = setup_logging()

# -----------------------
# Evaluation helper
# -----------------------


async def evaluate_metrics(output: str, context: str) -> dict:
    """Evaluate multiple metrics for a given LLM output.
    Metrics included: faithfulness, coherence, completeness.

    Args:
        output (str): The LLM-generated output to evaluate.
        context (str): The context used to generate the output.

    Returns:
        dict: A dictionary with metric names as keys and their evaluation results as values.

    """
    settings.openai.api_key = None
    logger.info(f"OpenAI key is not set: {settings.openai.api_key is None}")

    if not output.strip():
        logger.warning("Output is empty. Skipping evaluation.")
        return {
            "faithfulness": {"score": 0.0, "reason": "Empty output", "failed": True},
            "coherence": {"score": 0.0, "reason": "Empty output", "failed": True},
            "completeness": {"score": 0.0, "reason": "Empty output", "failed": True},
        }

    if not getattr(settings.openai, "api_key", None):
        logger.info("OpenAI API key not set. Skipping metrics evaluation.")
        return {
            "faithfulness": {"score": None, "reason": "Skipped – no API key", "failed": True},
            "coherence": {"score": None, "reason": "Skipped – no API key", "failed": True},
            "completeness": {"score": None, "reason": "Skipped – no API key", "failed": True},
        }

    judge_model = models.LiteLLMChatModel(
        model_name="gpt-4o",  # gpt-4o, gpt-5-mini
        api_key=settings.openai.api_key,
    )

    metric_configs = {
        "faithfulness": (
            (
                "You are an expert judge tasked with evaluating whether an AI-generated answer is "
                "faithful to the provided Substack excerpts."
            ),
            (
                "The OUTPUT must not introduce new information and beyond "
                "what is contained in the CONTEXT. "
                "All claims in the OUTPUT should be directly supported by the CONTEXT."
            ),
        ),
        "coherence": (
            (
                "You are an expert judge tasked with evaluating whether an AI-generated answer is "
                "logically coherent."
            ),
            "The answer should be well-structured, readable, and maintain consistent reasoning.",
        ),
        "completeness": (
            (
                "You are an expert judge tasked with evaluating whether an AI-generated answer "
                "covers all relevant aspects of the query."
            ),
            (
                "The answer should include all major points from the CONTEXT "
                "and address the user's "
                "query "
                "fully."
            ),
        ),
    }

    results = {}
    for name, (task_intro, eval_criteria) in metric_configs.items():
        try:
            metric = GEval(
                task_introduction=task_intro,
                evaluation_criteria=eval_criteria,
                model=judge_model,
                name=f"G-Eval {name.capitalize()}",
            )

            eval_input = f"""
            OUTPUT: {output}
            CONTEXT: {context}
            """

            score_result = await metric.ascore(eval_input)

            results[name] = {
                "score": score_result.value,
                "reason": score_result.reason,
                "failed": score_result.scoring_failed,
            }

        except Exception as e:
            logger.warning(f"G-Eval {name} failed: {e}")
            results[name] = {"score": 0.0, "reason": str(e), "failed": True}

    return results
