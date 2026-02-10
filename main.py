#!/usr/bin/env python3
"""
REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop

Main entry point for running attacks and evaluations.
"""

import argparse
import sys
from typing import List, Optional
import logging

from rebel import config


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_naive_attack(
    num_attacks: int,
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    use_hardcoded: bool = True,
    **kwargs
):
    """
    Run the naive leak@n attack approach.
    
    Args:
        num_attacks: Number of attacks to perform
        model_name: Custom model name (if not using hardcoded)
        tokenizer_name: Custom tokenizer name (if not using hardcoded)
        use_hardcoded: Whether to use hardcoded model configurations
    """
    from rebel.attacks.naive_attack import run_leak_attack
    
    logger = logging.getLogger(__name__)
    logger.info(f"Running naive attack (leak@{num_attacks})")
    
    if use_hardcoded:
        # Use hardcoded model configuration
        model_config = config.HARDCODED_MODELS.get("default", {})
        model_name = model_config.get("model_name")
        tokenizer_name = model_config.get("tokenizer_name")
        logger.info(f"Using hardcoded model: {model_name}")
    else:
        if not model_name or not tokenizer_name:
            raise ValueError("Model name and tokenizer name must be provided when not using hardcoded models")
        logger.info(f"Using custom model: {model_name}")
    
    # Call the actual attack implementation
    result = run_leak_attack(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        num_attacks=num_attacks,
        **kwargs
    )
    
    return result


def run_rebel_attack(
    num_attacks: int,
    top_k_list: List[int],
    mutations_list: List[str],
    model_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    use_hardcoded: bool = True,
    **kwargs
):
    """
    Run the REBEL evolutionary attack approach.
    
    Args:
        num_attacks: Number of attacks to perform
        top_k_list: List of top-k values for sampling
        mutations_list: List of mutation strategies to use
        model_name: Custom model name (if not using hardcoded)
        tokenizer_name: Custom tokenizer name (if not using hardcoded)
        use_hardcoded: Whether to use hardcoded model configurations
    """
    from rebel.attacks.rebel_attack import run_rebel
    
    logger = logging.getLogger(__name__)
    logger.info("Running REBEL evolutionary attack")
    
    if use_hardcoded:
        # Use hardcoded model configuration
        model_config = config.HARDCODED_MODELS.get("default", {})
        model_name = model_config.get("model_name")
        tokenizer_name = model_config.get("tokenizer_name")
        logger.info(f"Using hardcoded model: {model_name}")
    else:
        if not model_name or not tokenizer_name:
            raise ValueError("Model name and tokenizer name must be provided when not using hardcoded models")
        logger.info(f"Using custom model: {model_name}")
    
    # Call the actual attack implementation
    result = run_rebel(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        num_attacks=num_attacks,
        top_k_list=top_k_list,
        mutations_list=mutations_list,
        **kwargs
    )
    
    return result


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run naive attack with default parameters
  python main.py leak --num-attacks 100
  
  # Run naive attack with custom model
  python main.py leak --num-attacks 50 --model-name my-model --tokenizer-name my-tokenizer --no-hardcoded
  
  # Run REBEL evolutionary attack with defaults
  python main.py rebel
  
  # Run REBEL with custom parameters
  python main.py rebel --num-attacks 200 --top-k 5 10 20 --mutations synonym_replacement word_deletion
  
  # Run REBEL with custom model
  python main.py rebel --model-name my-model --tokenizer-name my-tokenizer --no-hardcoded
        """
    )
    
    # Common arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    
    # Subparsers for different attack modes
    subparsers = parser.add_subparsers(dest="mode", help="Attack mode to run", required=True)
    
    # Naive attack (leak@n) subparser
    leak_parser = subparsers.add_parser(
        "leak",
        help="Run naive leak@n attack",
        description="Run the naive attack approach with specified number of attacks"
    )
    leak_parser.add_argument(
        "--num-attacks",
        type=int,
        default=config.NUM_ATTACKS,
        help=f"Number of attacks to perform (default: {config.NUM_ATTACKS})"
    )
    leak_parser.add_argument(
        "--hardcoded",
        action="store_true",
        default=True,
        help="Use hardcoded model configurations (default: True)"
    )
    leak_parser.add_argument(
        "--no-hardcoded",
        dest="hardcoded",
        action="store_false",
        help="Use custom model names instead of hardcoded ones"
    )
    leak_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name from HuggingFace (required if --no-hardcoded)"
    )
    leak_parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Custom tokenizer name from HuggingFace (required if --no-hardcoded)"
    )
    
    # REBEL evolutionary attack subparser
    rebel_parser = subparsers.add_parser(
        "rebel",
        help="Run REBEL evolutionary attack",
        description="Run the REBEL evolutionary attack approach with configurable parameters"
    )
    rebel_parser.add_argument(
        "--num-attacks",
        type=int,
        default=config.NUM_ATTACKS,
        help=f"Number of attacks to perform (default: {config.NUM_ATTACKS})"
    )
    rebel_parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=config.TOP_K_LIST,
        help=f"Top-K values for sampling (default: {config.TOP_K_LIST})"
    )
    rebel_parser.add_argument(
        "--mutations",
        nargs="+",
        type=str,
        default=config.MUTATIONS_LIST,
        help=f"Mutation strategies to use (default: {config.MUTATIONS_LIST})"
    )
    rebel_parser.add_argument(
        "--hardcoded",
        action="store_true",
        default=True,
        help="Use hardcoded model configurations (default: True)"
    )
    rebel_parser.add_argument(
        "--no-hardcoded",
        dest="hardcoded",
        action="store_false",
        help="Use custom model names instead of hardcoded ones"
    )
    rebel_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name from HuggingFace (required if --no-hardcoded)"
    )
    rebel_parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Custom tokenizer name from HuggingFace (required if --no-hardcoded)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.mode == "leak":
            # Run naive attack
            result = run_naive_attack(
                num_attacks=args.num_attacks,
                model_name=args.model_name,
                tokenizer_name=args.tokenizer_name,
                use_hardcoded=args.hardcoded
            )
        elif args.mode == "rebel":
            # Run REBEL attack
            result = run_rebel_attack(
                num_attacks=args.num_attacks,
                top_k_list=args.top_k,
                mutations_list=args.mutations,
                model_name=args.model_name,
                tokenizer_name=args.tokenizer_name,
                use_hardcoded=args.hardcoded
            )
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        logger.info(f"Attack completed: {result}")
        
    except Exception as e:
        logger.error(f"Error running attack: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
