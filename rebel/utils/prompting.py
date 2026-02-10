"""
Utility functions for prompting and text manipulation.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def create_prompt(
    template: str,
    context: Dict[str, Any]
) -> str:
    """
    Create a prompt from a template and context.
    
    Args:
        template: Prompt template with placeholders
        context: Dictionary with values to fill in
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> prompt = create_prompt(
        ...     "Tell me about {topic}",
        ...     {"topic": "machine learning"}
        ... )
    """
    try:
        return template.format(**context)
    except KeyError as e:
        logger.error(f"Missing key in context: {e}")
        raise


def batch_prompts(
    prompts: List[str],
    batch_size: int
) -> List[List[str]]:
    """
    Batch prompts for efficient processing.
    
    Args:
        prompts: List of prompt strings
        batch_size: Size of each batch
        
    Returns:
        List of batched prompts
    """
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
