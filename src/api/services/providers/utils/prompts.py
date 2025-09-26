import opik

from src.api.models.api_models import SearchResult
from src.api.models.provider_models import ModelConfig

config = ModelConfig()

PROMPT = """
You are a skilled research assistant specialized in analyzing Substack newsletters.
Respond to the user’s query using the provided context from these articles,
that is retrieved from a vector database without relying on outside knowledge or assumptions.


### Output Rules:
- Write a detailed, structured answer using **Markdown** (headings, bullet points,
  short or long paragraphs as appropriate).
- Use up to **{tokens} tokens** without exceeding this limit.
- Only include facts from the provided context from the articles.
- Attribute each fact to the correct author(s) and source, and include **clickable links**.
- If the article author and feed author differ, mention both.
- There is no need to mention that you based your answer on the provided context.
- But if no relevant information exists, clearly state this and provide a fallback suggestion.
- At the very end, include a **funny quote** and wish the user a great day.

### Query:
{query}

### Context Articles:
{context_texts}

### Final Answer:
"""


# Create a new prompt
prompt = opik.Prompt(
    name="substack_research_assistant", prompt=PROMPT, metadata={"environment": "development"}
)


def build_research_prompt(
    contexts: list[SearchResult],
    query: str = "",
    tokens: int = config.max_completion_tokens,
) -> str:
    """Construct a research-focused LLM prompt using the given query
    and supporting context documents.

    The prompt enforces Markdown formatting, citations, and strict length guidance.

    Args:
        contexts (list[SearchResult]): List of context documents with metadata.
        query (str): The user's research query.
        tokens (int): Maximum number of tokens for the LLM response.

    Returns:
        str: The formatted prompt ready for LLM consumption.

    """
    # Join all retrieved contexts into a readable format
    context_texts = "\n\n".join(
        (
            f"- Feed Name: {r.feed_name}\n"
            f"  Article Title: {r.title}\n"
            f"  Article Author(s): {r.article_author}\n"
            f"  Feed Author: {r.feed_author}\n"
            f"  URL: {r.url}\n"
            f"  Snippet: {r.chunk_text}"
        )
        for r in contexts
    )

    return PROMPT.format(
        query=query,
        context_texts=context_texts,
        tokens=tokens,
    )
