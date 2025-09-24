"""
Module for generating multi-tool example queries and answers for dataset creation,
using LLM to generate synthetic dataset with realistic natural prompts.
"""

import random

from litellm import completion

from settings import settings
from src.dataset.prompts import create_multi_tool_prompt
from src.dataset.tools_description import TOOLS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_multi_tool_examples(count, start_id, use_hf_examples=True, num_examples=5):
    """
    Generate example queries and answers involving multiple tools. Randomly selects two
    tools; samples reasonable args for each tool; creates tool summary payload, readable
    descriptions for each function and its arguments; and passes payload to LLM to
    generate example.

    Args:
        count (int): Number of multi-tool examples to generate.
        start_id (int): Starting ID for the generated examples.
        use_hf_examples (bool, optional): Whether to use few-shot HuggingFace-style examples
        in the prompt. Defaults to True.
        num_examples (int, optional): Number of demonstration examples to include in the prompt.
        Defaults to 5.

    Returns:
        tuple: (examples, next_id)
            examples (list): List of generated example dicts, each with 'id', 'query', 'answers',
            and 'tools'.
            next_id (int): The next available ID after the last generated example.
    """
    examples = []
    tool_items = list(TOOLS.items())

    logger.info("Starting generation of %d multi-tool examples.", count)

    for i in range(count):
        selected = random.sample(tool_items, 2)
        tool_payload = []
        answers, tools_list = [], []

        logger.debug(
            "Example %d: Selected tools: %s", i + 1, [name for name, _ in selected]
        )

        for tool_name, tool_data in selected:
            args = {k: random.choice(v) for k, v in tool_data.get("args", {}).items()}
            answers.append({"name": tool_name, "arguments": args})
            tools_list.append(
                {
                    "name": tool_name,
                    "description": tool_data["description"],
                    "parameters": tool_data["parameters"],
                }
            )
            tool_payload.append(f"- {tool_name}({args}) ➜ {tool_data['description']}")
            logger.debug("Tool: %s, Args: %s", tool_name, args)

        prompt = create_multi_tool_prompt(
            tool_payload=tool_payload,
            use_hf_examples=use_hf_examples,
            num_examples=num_examples,
        )

        response = completion(
            model=settings.dataset.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            query = response["choices"][0]["message"]["content"].strip()
            examples.append(
                {
                    "id": start_id,
                    "query": query,
                    "answers": answers,
                    "tools": tools_list,
                }
            )
            logger.info("Generated example with id %d", start_id)
            start_id += 1
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Skipping due to malformed LLM response: %s", e)

    logger.info("Finished generating examples. Total: %d", len(examples))
    return examples, start_id


def main() -> None:
    "Run script"
    generated_examples, _ = generate_multi_tool_examples(10, 1)
    logger.info("Generated examples: %s", generated_examples)


if __name__ == "__main__":
    main()
