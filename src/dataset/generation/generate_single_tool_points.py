import random

from src.dataset.utils import render_template
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_single_tool_examples(tool_name, tool_data, count, start_id):
    """
    Generate example queries and answers for a single tool.

    Args:
        tool_name (str): The name of the tool.
        tool_data (dict): Dictionary containing tool metadata, including 'args', 'templates',
        'description', and 'parameters'.
        count (int): Number of examples to generate.
        start_id (int): Starting ID for the generated examples.

    Returns:
        tuple: (examples, next_id)
            examples (list): List of generated example dicts, each with 'id', 'query', 'answers',
            and 'tools'.
            next_id (int): The next available ID after the last generated example.
    """
    logger.info(
        "Generating %d examples for tool '%s' starting from ID %d",
        count,
        tool_name,
        start_id,
    )
    examples = []
    for i in range(count):
        args = {k: random.choice(v) for k, v in tool_data.get("args", {}).items()}
        template = random.choice(tool_data["templates"])
        query = render_template(template, args)
        answers = [{"name": tool_name, "arguments": args}]
        tools = [
            {
                "name": tool_name,
                "description": tool_data["description"],
                "parameters": tool_data["parameters"],
            }
        ]
        logger.debug(
            "Example %d: args=%s, template='%s', query='%s'",
            i + 1,
            args,
            template,
            query,
        )
        examples.append(
            {"id": start_id, "query": query, "answers": answers, "tools": tools}
        )
        start_id += 1
    logger.info("Generated %d examples for tool '%s'", len(examples), tool_name)
    return examples, start_id
