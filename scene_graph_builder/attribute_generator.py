import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass

from api import StreamGenerator
from utils import JSONParser

logger = logging.getLogger(__name__)


@dataclass
class AttributeGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "attributes.json"


class AttributeGenerator:
    SUPPORTED_REASONING_MODELS = [""]

    def __init__(self, config: AttributeGenerationConfig):
        self.config = config
        self.existing_data = self._load_existing_data()

    def _load_existing_data(self) -> List[Dict[str, Any]]:
        """Load existing data from output file if it exists"""
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_data(self, data: List[Dict[str, Any]]):
        """Save data to output file with atomic write"""
        temp_file = f"{self.config.output_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(temp_file).replace(self.config.output_file)

    async def _generate_attributes(
            self,
            prompts: List[str],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Generate attributes from prompts using LLM"""
        generator = StreamGenerator(
            model_name=self.config.model_name,
            api_keys=self.config.api_keys,
            max_concurrent_per_key=self.config.max_concurrent_per_key,
            max_retries=self.config.max_retries,
            rational=self.config.model_name in self.SUPPORTED_REASONING_MODELS,
        )

        results = []
       
        async for result in generator.generate_stream(prompts, self.config.system_prompt, validate_func):
            if result is not None:
                results.append(result)
        return results

    def _validate_attribute_response(self, response: str) -> Union[Dict[str, Dict[str, List[str]]], bool]:
        """Validate attribute format (concepts with values)"""
        parsed = JSONParser.parse(response)
        if not isinstance(parsed, dict):
            return False
        required_keys = {"object_id", "object_name", "attributes"}
        if not all(k in parsed for k in required_keys):
            return False

        if not isinstance(parsed["attributes"], dict):
            return False

        for concept, values in parsed["attributes"].items():
            if not isinstance(concept, str):
                return False
            if not isinstance(values, list):
                return False
            for value in values:
                if not isinstance(value, str):
                    return False
        return parsed

    def _create_attribute_prompts(
            self,
            objects: List[Dict[str, Any]],
            concepts_per_object: int,
            values_per_concept: int) -> List[str]:
        return [
    f"""Generate a JSON response with exactly {concepts_per_object} VISUALIZABLE attribute concepts and {values_per_concept} VISUALIZABLE values per concept for the object '{obj['name']}'.

Response Structure:
{{
    "object_id": {obj['id']},
    "object_name": "{obj['name']}",
    "attributes": {{
        "concept1": ["value1", "value2", ...],  # Exactly {values_per_concept} items
        "concept2": ["value1", "value2", ...],
        ...
    }}  # Exactly {concepts_per_object} concepts
}}

Requirements:
1. Attribute concepts must be:
   - Instantly visually recognizable and concrete
   - Directly related to the physical appearance or form of the object
   - Simple, fundamental, and not abstract
2. Values must be:
   - Single words when possible
   - Clearly distinct from each other
   - Common, easily understood, and visually obvious
3. DO NOT include abstract or ambiguous adjectives such as "relaxed", "alert", "matte", or any similar terms that do not translate into clear visual features.
4. Strictly avoid underscores (_) in the output. Always use spaces instead of underscores.

Good Example (Simple and highly visual):
{{
    "object_name": "painting",
    "attributes": {{
        "color": ["red", "blue", "green", "yellow", "black"],
        "style": ["abstract", "realistic", "impressionist", "cubist", "surrealist"],
        "size": ["small", "medium", "large", "tiny", "huge"]
    }}
}}

Bad Example (Too complex and not always visually obvious):
{{
    "object_name": "painting",
    "attributes": {{
        "color_palette": ["monochromatic", "warm tones", "cool tones", "pastel", "vibrant"],
        "brushwork_style": ["smooth/blended", "impasto (thick)", "pointillism (dots)", "cross-hatching", "loose/expressive"],
        "subject_matter": ["landscape", "portrait", "still life", "abstract", "historical"],
        "composition": ["symmetrical", "rule of thirds", "central focus", "diagonal", "minimalist"],
        "lighting_effect": ["high contrast", "soft diffused", "dramatic chiaroscuro", "backlit", "uniform flat"]
    }}
}}

Object to analyze: {obj['name']}"""
    for obj in objects
]




    async def generate_attributes(
            self,
            input_data: List[Dict[str, Any]],
            concepts_per_object: int = 5,
            values_per_concept: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate attributes (concepts with values) for each object"""
        logger.info(f"Generating {concepts_per_object} attributes with {values_per_concept} values each per object")

        # Extract all objects from categories
        objects = []
        for category in input_data:
            for obj in category["objects"]:
                objects.append({
                    "id": obj["id"],
                    "name": obj["name"],
                    "category_id": category["category_id"],
                    "category_name": category["category_name"]
                })

        prompts = self._create_attribute_prompts(objects, concepts_per_object, values_per_concept)
        results = await self._generate_attributes(prompts, self._validate_attribute_response)

        output_data = self._organize_results(input_data, results)
        self._save_data(output_data)

        return output_data

    def _organize_results(
            self,
            input_data: List[Dict[str, Any]],
            attribute_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Organize attribute results into the original category structure"""
        attribute_map = {res["object_name"]: res["attributes"] for res in attribute_results}

        output_data = []
        for category in input_data:
            category_entry = {
                "category_id": category["category_id"],
                "category_name": category["category_name"],
                "objects": []
            }

            for obj in category["objects"]:
                obj_entry = {
                    "id": obj["id"],
                    "name": obj["name"],
                    "attributes": {}
                }

                if obj["name"] in attribute_map:
                    obj_entry["attributes"] = attribute_map[obj["name"]]
              
                category_entry["objects"].append(obj_entry)
        
            output_data.append(category_entry)
        return output_data