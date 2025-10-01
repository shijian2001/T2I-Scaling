
import os
import argparse
import asyncio
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any
import sys
import os
from pathlib import Path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
from promptqa_generator import PromptGenerator, PromptGenerationConfig
print(sys.path)
from utils import setup_logger

def load_scene_graphs(scene_graph_file: str, difficulty: str) -> List[Dict]:
    try:
        with open(scene_graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            graphs = data.get(difficulty, [])
            if not isinstance(graphs, list):
                raise ValueError(f"Expected list for difficulty {difficulty}, got {type(graphs)}")
            return graphs
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid scene graph file: {e}")

def load_config(config_file: str, d_min: float, d_max: float) -> Dict[str, Any]:
    if d_min > d_max:
        raise ValueError(f"d_min must be less or equal than d_max, got d_min={d_min}, d_max={d_max}")
    if d_min < 0 or d_max > 10:
        raise ValueError(f"Difficulty range must be between 0 and 10, got {d_min}-{d_max}")
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Config should be a YAML dictionary")
            
            def replace_placeholders(value):
                if isinstance(value, str):
                    return value.replace("{d_min}", str(d_min)).replace("{d_max}", str(d_max))
                return value
            
            config = {k: replace_placeholders(v) for k, v in config.items()}
            
            if "output_file" in config:
                dir_path = os.path.dirname(config["output_file"])
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Created directory: {dir_path}")
            
            return config
            
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid config file: {e}")
    except OSError as e:
        raise ValueError(f"Failed to create directories: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts from scene graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        help="Path to the main YAML configuration file (configs/prompts.yaml)"
    )
    parser.add_argument(
        "--d-min",
        type=int,
        required=True,
        help="Minimum difficulty level (0-10)"
    )
    parser.add_argument(
        "--d-max",
        type=int,
        required=True,
        help="Maximum difficulty level (0-10)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    setup_logger(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Processing difficulty range: {args.d_min}-{args.d_max}")
        
        main_config = load_config(args.config_file, args.d_min, args.d_max)

        keys_path = main_config.get("keys_path", "configs/keys.yaml")
        with open(keys_path, "r", encoding="utf-8") as f:
            api_keys = yaml.safe_load(f).get("keys", [])

        input_file = main_config.get("input_file")
        if input_file is None:
            raise ValueError("input_file not specified in the config file")
        output_file = main_config.get("output_file")
        if output_file is None:
            raise ValueError("output_file not specified in the config file")
        difficulty_str = main_config.get("difficulty", f"difficulty_{args.d_min}_{args.d_max}")
        model_name = main_config.get("model_name", "")
        max_concurrent_per_key = main_config.get("max_concurrent_per_key", 300)
        max_retries = main_config.get("max_retries", 5)
        system_prompt = main_config.get("system_prompt", "")

        config = PromptGenerationConfig(
            model_name=model_name,
            api_keys=api_keys,
            system_prompt=system_prompt,
            output_file=output_file,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries
        )

        generator = PromptGenerator(config)
        scene_graphs = load_scene_graphs(input_file, difficulty_str)

        async def run_generation():
            return await generator.generate_prompts(scene_graphs, difficulty_str)

        results = asyncio.run(run_generation())

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generator.existing_data, f, indent=2)

        logger.info(f"Prompt generation for {difficulty_str} completed. Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
