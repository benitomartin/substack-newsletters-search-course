from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam


def build_messages(
    prompt: str,
) -> list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]:
    """Build a list of messages for the OpenAI chat API.

    Args:
        prompt (str): The user prompt.

    Returns:
        list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]: A list of messages.

    """
    return [
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]
