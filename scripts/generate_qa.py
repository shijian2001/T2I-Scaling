import json
from collections import defaultdict
from typing import Dict, Any
import argparse
import yaml
import logging
from pathlib import Path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
import sys
sys.path.append(str(root_dir))
from promptqa_generator import process_scene_graphs
from utils import setup_logger
import os



def load_config(config_file: str, d_min: float, d_max: float) -> Dict[str, Any]:
    if d_min > d_max:
        raise ValueError(f"d_min must be less than or equal d_max, got d_min={d_min}, d_max={d_max}")
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
            required_fields = ["input_file", "output_file"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")

            if "output_file" in config:
                dir_path = os.path.dirname(config["output_file"])
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    logging.info(f"Created directory: {dir_path}")

            return config

    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid config file: {e}")
    except OSError as e:
        raise ValueError(f"Failed to create directories: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from scene graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        help="Path to the main YAML configuration file (configs/qa.yaml)"
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
        print('*' * 120)
        print(main_config)

        process_scene_graphs(main_config['input_file'], main_config['output_file'])



        logger.info(
            f"Successfully generated QA pairs for {args.d_min}-{args.d_max} scene graphs\n"
        )

    except Exception as e:
        logger.error(f"QA generation failed: {e}")
        raise

# 使用示例
if __name__ == "__main__":
    main()
