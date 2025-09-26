import os

import gradio as gr
import markdown
import requests
import yaml
from dotenv import load_dotenv

try:
    from src.api.models.provider_models import MODEL_REGISTRY
except ImportError as e:
    raise ImportError(
        "Could not import MODEL_REGISTRY from src.api.models.provider_models. "
        "Check the path and file existence."
    ) from e

# Initialize environment variables
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
API_BASE_URL = f"{BACKEND_URL}/search"


# Load feeds from YAML
def load_feeds():
    """Load feeds from the YAML configuration file.
    Returns:
        list: List of feeds with their details.
    """
    feeds_path = os.path.join(os.path.dirname(__file__), "../src/configs/feeds_rss.yaml")
    with open(feeds_path) as f:
        feeds_yaml = yaml.safe_load(f)
    return feeds_yaml.get("feeds", [])


feeds = load_feeds()
feed_names = [f["name"] for f in feeds]
feed_authors = [f["author"] for f in feeds]


# -----------------------
# API helpers
# -----------------------
def fetch_unique_titles(payload):
    """
    Fetch unique article titles based on the search criteria.

    Args:
        payload (dict): The search criteria including query_text, feed_author,
                        feed_name, limit, and optional title_keywords.
    Returns:
        list: A list of articles matching the criteria.
    Raises:
        Exception: If the API request fails.
    """
    try:
        resp = requests.post(f"{API_BASE_URL}/unique-titles", json=payload)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        raise Exception(f"Failed to fetch titles: {str(e)}") from e


def call_ai(payload, streaming=True):
    """ "
    Call the AI endpoint with the given payload.
    Args:
        payload (dict): The payload to send to the AI endpoint.
        streaming (bool): Whether to use streaming or non-streaming endpoint.
    Yields:
        tuple: A tuple containing the type of response and the response text.
    """
    endpoint = f"{API_BASE_URL}/ask/stream" if streaming else f"{API_BASE_URL}/ask"
    answer_text = ""
    try:
        if streaming:
            with requests.post(endpoint, json=payload, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    if chunk.startswith("__model_used__:"):
                        yield "model", chunk.replace("__model_used__:", "").strip()
                    elif chunk.startswith("__error__"):
                        yield "error", "Request failed. Please try again later."
                        break
                    elif chunk.startswith("__truncated__"):
                        yield "truncated", "AI response truncated due to token limit."
                    else:
                        answer_text += chunk
                        yield "text", answer_text
        else:
            resp = requests.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer_text = data.get("answer", "")
            yield "text", answer_text
            if data.get("finish_reason") == "length":
                yield "truncated", "AI response truncated due to token limit."
    except Exception as e:
        yield "error", f"Request failed: {str(e)}"


def get_models_for_provider(provider):
    """
    Get available models for a provider

    Args:
        provider (str): The name of the provider (e.g., "openrouter", "openai")
    Returns:
        list: List of model names available for the provider
    """
    provider_key = provider.lower()
    try:
        config = MODEL_REGISTRY.get_config(provider_key)
        return (
            ["Automatic Model Selection (Model Routing)"]
            + ([config.primary_model] if config.primary_model else [])
            + list(config.candidate_models)
        )
    except Exception:
        return ["Automatic Model Selection (Model Routing)"]


# -----------------------
# Gradio interface functions
# -----------------------
def handle_search_articles(query_text, feed_name, feed_author, title_keywords, limit):
    """
    Handle article search

    Args:
        query_text (str): The text to search for in article titles.
        feed_name (str): The name of the feed to filter articles by.
        feed_author (str): The author of the feed to filter articles by.
        title_keywords (str): Keywords to search for in article titles.
        limit (int): The maximum number of articles to return.
    Returns:
        str: HTML formatted string of search results or error message.
    Raises:
        Exception: If the API request fails.
    """
    if not query_text.strip():
        return "Please enter a query text."

    payload = {
        "query_text": query_text.strip().lower(),
        "feed_author": feed_author.strip() if feed_author else "",
        "feed_name": feed_name.strip() if feed_name else "",
        "limit": limit,
        "title_keywords": title_keywords.strip().lower() if title_keywords else None,
    }

    try:
        results = fetch_unique_titles(payload)
        if not results:
            return "No results found."

        html_output = ""
        for item in results:
            html_output += (
                f"<div style='background-color:#F0F8FF; padding:20px; "
                f"border-radius:10px; font-size:18px; margin-bottom:15px;'>\n"
                f"    <h2 style='font-size:22px; color:#1f4e79; margin-top:0;'>"
                f"{item.get('title', 'No title')}</h2>\n"
                f"    <p style='margin:5px 0;'>"
                f"<b>Newsletter:</b> {item.get('feed_name', 'N/A')}"
                f"</p>\n"
                f"    <p style='margin:5px 0;'>"
                f"<b>Author:</b> {item.get('feed_author', 'N/A')}"
                f"</p>\n"
                f"    <p style='margin:5px 0;'><b>Article Authors:</b> "
                f"{', '.join(item.get('article_author') or ['N/A'])}</p>\n"
                f"    <p style='margin:5px 0;'><b>URL:</b> "
                f"<a href='{item.get('url', '#')}' target='_blank' style='color:#0066cc;'>"
                f"{item.get('url', 'No URL')}</a></p>\n"
                f"</div>\n"
            )
        return html_output

    except Exception as e:
        return f"<div style='color:red; padding:10px;'>Error: {str(e)}</div>"


def handle_ai_question_streaming(
    query_text,
    feed_name,
    feed_author,
    limit,
    provider,
    model,
):
    """
    Handle AI question with streaming

    Args:
        query_text (str): The question to ask the AI.
        feed_name (str): The name of the feed to filter articles by.
        feed_author (str): The author of the feed to filter articles by.
        limit (int): The maximum number of articles to consider.
        provider (str): The LLM provider to use.
        model (str): The specific model to use from the provider.
    Yields:
        tuple: (HTML formatted answer string, model info string)
    """
    if not query_text.strip():
        yield "Please enter a query text.", ""
        return

    if not provider or not model:
        yield "Please select provider and model.", ""
        return

    payload = {
        "query_text": query_text.strip().lower(),
        "feed_author": feed_author.strip() if feed_author else "",
        "feed_name": feed_name.strip() if feed_name else "",
        "limit": limit,
        "provider": provider.lower(),
    }

    if model != "Automatic Model Selection (Model Routing)":
        payload["model"] = model

    try:
        answer_html = ""
        model_info = f"Provider: {provider}"

        for _, (event_type, content) in enumerate(call_ai(payload, streaming=True)):
            if event_type == "text":
                # Convert markdown to HTML
                html_content = markdown.markdown(content, extensions=["tables"])
                answer_html = (
                    f"\n"
                    f"<div style='background-color:#E8F0FE; "
                    f"padding:15px; border-radius:10px; font-size:16px;'>\n"
                    f"    {html_content}\n"
                    f"</div>\n"
                )
                yield answer_html, model_info

            elif event_type == "model":
                model_info = f"Provider: {provider} | Model: {content}"
                yield answer_html, model_info

            elif event_type == "truncated":
                answer_html += (
                    f"<div style='color:#ff6600; padding:10px; font-weight:bold;'>⚠️ {content}</div>"
                )
                yield answer_html, model_info

            elif event_type == "error":
                error_html = (
                    f"<div style='color:red; padding:10px; font-weight:bold;'>❌ {content}</div>"
                )
                yield error_html, model_info
                break

    except Exception as e:
        error_html = f"<div style='color:red; padding:10px;'>Error: {str(e)}</div>"
        yield error_html, model_info


def handle_ai_question_non_streaming(query_text, feed_name, feed_author, limit, provider, model):
    """
    Handle AI question without streaming

    Args:
        query_text (str): The question to ask the AI.
        feed_name (str): The name of the feed to filter articles by.
        feed_author (str): The author of the feed to filter articles by.
        limit (int): The maximum number of articles to consider.
        provider (str): The LLM provider to use.
        model (str): The specific model to use from the provider.

    Returns:
        tuple: (HTML formatted answer string, model info string)
    """
    if not query_text.strip():
        return "Please enter a query text.", ""

    if not provider or not model:
        return "Please select provider and model.", ""

    payload = {
        "query_text": query_text.strip().lower(),
        "feed_author": feed_author.strip() if feed_author else "",
        "feed_name": feed_name.strip() if feed_name else "",
        "limit": limit,
        "provider": provider.lower(),
    }

    if model != "Automatic Model Selection (Model Routing)":
        payload["model"] = model

    try:
        answer_html = ""
        model_info = f"Provider: {provider}"

        for event_type, content in call_ai(payload, streaming=False):
            if event_type == "text":
                html_content = markdown.markdown(content, extensions=["tables"])
                answer_html = (
                    "<div style='background-color:#E8F0FE; "
                    "padding:15px; border-radius:10px; font-size:16px;'>\n"
                    f"{html_content}\n"
                    "</div>\n"
                )
            elif event_type == "model":
                model_info = f"Provider: {provider} | Model: {content}"
            elif event_type == "truncated":
                answer_html += (
                    f"<div style='color:#ff6600; padding:10px; font-weight:bold;'>⚠️ {content}</div>"
                )
            elif event_type == "error":
                return (
                    f"<div style='color:red; padding:10px; font-weight:bold;'>❌ {content}</div>",
                    model_info,
                )

        return answer_html, model_info

    except Exception as e:
        return (
            f"<div style='color:red; padding:10px;'>Error: {str(e)}</div>",
            f"Provider: {provider}",
        )


def update_model_choices(provider):
    """
    Update model choices based on selected provider
    Args:
        provider (str): The selected LLM provider
    Returns:
        gr.Dropdown: Updated model dropdown component
    """
    models = get_models_for_provider(provider)
    return gr.Dropdown(choices=models, value=models[0] if models else "")


# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks(title="Substack Articles LLM Engine", theme=gr.themes.Soft()) as demo:
    # Header
    gr.HTML(
        "<div style='background-color:#ff6719; padding:20px; border-radius:12px; "
        "text-align:center; margin-bottom:20px;'>\n"
        "    <h1 style='color:white; font-size:42px; font-family:serif; margin:0;'>\n"
        "        📰 Substack Articles LLM Engine\n"
        "    </h1>\n"
        "</div>\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Search Mode Selection
            gr.Markdown("## 🔍 Select Search Mode")
            search_type = gr.Radio(
                choices=["Search Articles", "Ask the AI"],
                value="Search Articles",
                label="Search Mode",
                info="Choose between searching for articles or asking AI questions",
            )

            # Common filters
            gr.Markdown("### Filters")
            query_text = gr.Textbox(label="Query", placeholder="Type your query here...", lines=3)
            feed_author = gr.Dropdown(
                choices=[""] + feed_authors, label="Author (optional)", value=""
            )
            feed_name = gr.Dropdown(
                choices=[""] + feed_names, label="Newsletter (optional)", value=""
            )

            # Conditional fields based on search type
            title_keywords = gr.Textbox(
                label="Title Keywords (optional)",
                placeholder="Filter by words in the title",
                visible=True,
            )

            limit = gr.Slider(
                minimum=1, maximum=20, step=1, label="Number of results", value=5, visible=True
            )

            # LLM Options (only visible for AI mode)
            with gr.Group(visible=False) as llm_options:
                gr.Markdown("### ⚙️ LLM Options")
                provider = gr.Dropdown(
                    choices=["OpenRouter", "HuggingFace", "OpenAI"],
                    label="Select LLM Provider",
                    value="OpenRouter",
                )
                model = gr.Dropdown(
                    choices=get_models_for_provider("OpenRouter"),
                    label="Select Model",
                    value="Automatic Model Selection (Model Routing)",
                )
                streaming_mode = gr.Radio(
                    choices=["Streaming", "Non-Streaming"],
                    value="Streaming",
                    label="Answer Mode",
                    info="Streaming shows results as they're generated",
                )

            # Submit button
            submit_btn = gr.Button("🔎 Search / Ask AI", variant="primary", size="lg")

        with gr.Column(scale=2):
            # Output area
            output_html = gr.HTML(label="Results")
            model_info = gr.HTML(visible=False)

    # Event handlers
    def toggle_visibility(search_type):
        """
        Toggle visibility of components based on search type

        Args:
            search_type (str): The selected search type
        Returns:
            tuple: Visibility states for (llm_options, title_keywords, model_info)
        """

        show_title_keywords = search_type == "Search Articles"
        show_llm_options = search_type == "Ask the AI"
        show_model_info = search_type == "Ask the AI"
        show_limit_slider = search_type == "Search Articles"

        return (
            gr.Group(visible=show_llm_options),  # llm_options
            gr.Textbox(visible=show_title_keywords),  # title_keywords
            gr.HTML(visible=show_model_info),  # model_info
            gr.Slider(visible=show_limit_slider),  # limit
        )

    search_type.change(
        fn=toggle_visibility,
        inputs=[search_type],
        outputs=[llm_options, title_keywords, model_info, limit],
    )

    # Update model dropdown when provider changes
    provider.change(fn=update_model_choices, inputs=[provider], outputs=[model])

    # Unified submission handler
    def handle_submission(
        search_type,
        streaming_mode,
        query_text,
        feed_name,
        feed_author,
        title_keywords,
        limit,
        provider,
        model,
    ):
        """
        Handle submission based on search type and streaming mode
        Args:
            search_type (str): The selected search type
            streaming_mode (str): The selected streaming mode
            query_text (str): The query text
            feed_name (str): The selected feed name
            feed_author (str): The selected feed author
            title_keywords (str): The title keywords (if applicable)
            limit (int): The number of results to return
            provider (str): The selected LLM provider (if applicable)
            model (str): The selected model (if applicable)
        Returns:
            tuple: (HTML formatted answer string, model info string)
        """
        if search_type == "Search Articles":
            result = handle_search_articles(
                query_text, feed_name, feed_author, title_keywords, limit
            )
            return result, ""  # Always return two values
        else:  # Ask the AI
            if streaming_mode == "Non-Streaming":
                return handle_ai_question_non_streaming(
                    query_text, feed_name, feed_author, limit, provider, model
                )
            else:
                # For streaming, we'll use a separate handler
                return "", ""

    # Streaming handler
    def handle_streaming_submission(
        search_type,
        streaming_mode,
        query_text,
        feed_name,
        feed_author,
        title_keywords,
        limit,
        provider,
        model,
    ):
        """
        Handle submission with streaming support
        Args:
            search_type (str): The selected search type
            streaming_mode (str): The selected streaming mode
            query_text (str): The query text
            feed_name (str): The selected feed name
            feed_author (str): The selected feed author
            title_keywords (str): The title keywords (if applicable)
            limit (int): The number of results to return
            provider (str): The selected LLM provider (if applicable)
            model (str): The selected model (if applicable)
        Yields:
            tuple: (HTML formatted answer string, model info string)
        """
        if search_type == "Ask the AI" and streaming_mode == "Streaming":
            yield from handle_ai_question_streaming(
                query_text, feed_name, feed_author, limit, provider, model
            )
        else:
            # For non-streaming cases, just return the regular result
            if search_type == "Search Articles":
                result = handle_search_articles(
                    query_text, feed_name, feed_author, title_keywords, limit
                )
                yield result, ""
            else:
                result_html, model_info_text = handle_ai_question_non_streaming(
                    query_text, feed_name, feed_author, limit, provider, model
                )
                yield result_html, model_info_text

    # Single click handler that routes based on mode
    submit_btn.click(
        fn=handle_streaming_submission,
        inputs=[
            search_type,
            streaming_mode,
            query_text,
            feed_name,
            feed_author,
            title_keywords,
            limit,
            provider,
            model,
        ],
        outputs=[output_html, model_info],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch()
