import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass
import os
import re
from pathlib import Path
import sys

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from api import StreamGenerator
from utils import JSONParser

logger = logging.getLogger(__name__)


@dataclass
class PromptGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "prompts.json"


class PromptGenerator:
    SUPPORTED_REASONING_MODELS = [""]

    def __init__(self, config: PromptGenerationConfig):
        self.config = config
        self.processed_ids = set()
        self.existing_data = self._load_existing_data()

    def _load_existing_data(self) -> Dict[str, Any]:
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "difficulty" in data:
                    for item in data["difficulty"]:
                        if "id" in item:
                            self.processed_ids.add(str(item["id"]))
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_data(self, data: Dict[str, Any]):
        temp_file = f"{self.config.output_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(temp_file).replace(self.config.output_file)

    async def _generate_prompts(
            self,
            prompts_with_index: List[tuple],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[tuple]:
        generator = StreamGenerator(
            model_name=self.config.model_name,
            api_keys=self.config.api_keys,
            max_concurrent_per_key=self.config.max_concurrent_per_key,
            max_retries=self.config.max_retries,
            rational=self.config.model_name in self.SUPPORTED_REASONING_MODELS,
        )

        results = []
        async for index, result in generator.generate_stream_with_index(
                prompts_with_index, self.config.system_prompt, validate_func
        ):
            if result is not None:
                results.append((index, result))
        results.sort(key=lambda x: x[0])
        return results

    def _validate_prompt(self, response: str) -> Union[Dict[str, Any], bool]:
        parsed = JSONParser.parse(response)
        if isinstance(parsed, dict) and "prompt" in parsed:
            return parsed
        return False

    def _remove_uuid_like(self, text):
        uuid_pattern = re.compile(r'\\u[0-9a-fA-F]{4}')
        return uuid_pattern.sub('', text)

    def _preprocess_scene_graph(self, scene_graph: Dict) -> Dict:
        processed_graph = scene_graph.copy()

        processed_graph.pop("computed_diff", None)

        name_counts = {}
        for obj in processed_graph.get("objects", []):
            name = obj["name"]
            name_counts[name] = name_counts.get(name, 0) + 1

        name_indices = {}
        for i, obj in enumerate(processed_graph.get("objects", [])):
            name = obj["name"]
            if name_counts[name] > 1:
                idx = name_indices.get(name, 0)
                name_indices[name] = idx + 1

                obj["temp_id"] = f"{idx + 1}"
                obj["display_name"] = name
            else:
                obj["display_name"] = name

        return processed_graph

    def _create_prompts(self, scene_graphs: List[Dict]) -> List[tuple]:
        if not scene_graphs:
            return []

        generated_prompts = []
        for index, scene_graph in enumerate(scene_graphs):
            if not scene_graph or "scene_graph" not in scene_graph:
                continue

            if "id" in scene_graph and str(scene_graph["id"]) in self.processed_ids:
                continue

            processed_graph = self._preprocess_scene_graph(scene_graph["scene_graph"])
            obj_id_to_name = {}
            obj_id_to_temp_id = {}
            for obj in processed_graph.get("objects", []):
                if "id" in obj:
                    obj_id_to_name[obj["id"]] = obj["display_name"]
                    if "temp_id" in obj:
                        obj_id_to_temp_id[obj["id"]] = obj["temp_id"]

            for rel in processed_graph.get("relations", []):
                if "subject_id" in rel and rel["subject_id"] in obj_id_to_name:
                    rel["subject_display"] = obj_id_to_name[rel["subject_id"]]
                    if rel["subject_id"] in obj_id_to_temp_id:
                        rel["subject_temp_id"] = obj_id_to_temp_id[rel["subject_id"]]

                if "object_id" in rel and rel["object_id"] in obj_id_to_name:
                    rel["object_display"] = obj_id_to_name[rel["object_id"]]
                    if rel["object_id"] in obj_id_to_temp_id:
                        rel["object_temp_id"] = obj_id_to_temp_id[rel["object_id"]]

            scene_graph_json = json.dumps(processed_graph, ensure_ascii=False, indent=2)

            prompt = f"""Given the following scene description components in JSON format:
{scene_graph_json}

Generate a single, coherent description that EXACTLY represents all given elements:

STRICT REQUIREMENTS:
1. ONLY mention relationships that are EXPLICITLY defined in the "relations" array
2. DO NOT create any new relationships between objects that don't exist in the "relations" array
3. DO NOT use prepositions like "in", "on", "with", "near", etc. unless they are the EXACT relation words in the "relations" array
4. NEVER group objects together unless they have an explicit relationship in the "relations" array
5. List all objects separately if they don't have relationships with other objects
6. Include EVERY object with its EXACT attributes as specified in the scene graph
7. For objects with the same name, use their attributes or explicit relationships to distinguish them
8. DO NOT use artificial identifiers like "#1" or "#2" in the final description
9. MUST use the ORIGINAL wording for ALL elements (objects/relations/attributes)
10. MUST NOT add any information not present in the scene graph
11. 12. List every object instance explicitly. For identical objects, state their count (e.g., 'two identical apples'). For similar objects with differences, describe each one's attributes (e.g., 'a red apple and a green apple')

EXAMPLE OF WHAT NOT TO DO:
Scene graph has: "desert with shrubs", "eagle", "rocket", but NO relations between them
BAD: "In a desert with shrubs, there is an eagle and a rocket" (incorrectly implies eagle and rocket are in the desert)
GOOD: "A desert with shrubs, an eagle, and a rocket" (lists objects separately without implying relationships)

Scene graph has: "truck", "pizza with olive oil sauce", "truck with a flat cabin type", "motorcycle with a boxy shape", "orange juice", "meadow", "monitor", but NO relations between them
BAD: "There is a truck, a pizza with olive oil sauce, a truck with a flat cabin type, a motorcycle with a boxy shape, orange juice, a meadow, and a monitor." (incorrectly implies all these objects are in the same context without clarification of relationships)
GOOD: "There are two trucks, one with a flat cabin type, a pizza with olive oil sauce, a motorcycle with a boxy shape, orange juice, a meadow, and a monitor." (clearly separates the objects and their attributes)

Scene graph has: "compact truck", "green truck", "yellow printer", "teenager with sandals", "doctor with short hair", but NO relations between them
BAD: "A compact truck, a green truck, a yellow printer, a teenager with sandals, and a doctor with short hair." (incorrectly lists objects without specifying relationships)
GOOD: "There are two trucks: a compact truck and a green truck, a yellow printer, a teenager with sandals, and a doctor with short hair." (clearly distinguishes between the trucks and other objects)


EXAMPLE OF CORRECT FORMATTING:
Scene graph has: "dog" with attribute "white", "dog" with attribute "yellow", "hamburger", and relation "dog (white) eating hamburger"
GOOD: "A white dog is eating a hamburger, and a yellow dog is present."

Bad Example (without specifying the quantity of the same objects):
"A fire hydrant, a slim router, an adult, a waterfall, a cow, a manhole cover, a waterfall with a veil shape, and another manhole cover."
Good Example (specifying the quantity of the same objects):
"A fire hydrant, a slim router, an adult, two distinct waterfalls, one being standard and the other featuring a veil-like shape, a cow, and two standard manhole covers."

Bad Example (without specifying the quantity of the same objects):
"A truck, a pizza with olive oil sauce, a truck with a flat cabin type, a motorcycle with a boxy shape, orange juice, a meadow, and a monitor."
Good Example (specifying the quantity of the same objects):
"There are two trucks where a truck with a flat cabin type, a pizza with olive oil sauce, , a motorcycle with a boxy shape, orange juice, a meadow, and a monitor."

Bad Example (without specifying the quantity of the same objects):
"A compact truck, a green truck, a yellow printer, a teenager with sandals, and a doctor with short hair."
Good Example (specifying the quantity of the same objects):
"There are two truck, a compact truck, a green truck, a yellow printer, a teenager with sandals, and a doctor with short hair."



Output ONLY this JSON format with NO additional text:
{{"prompt": "Generated description here"}}
"""

            clean_prompt = self._remove_uuid_like(prompt)
            generated_prompts.append((index, clean_prompt))

        return generated_prompts

    async def generate_prompts(self, scene_graphs: List[Dict], difficulty_str: str) -> List[Dict[str, Any]]:
        logger.info(f"Generating prompts for {len(scene_graphs)} scene graphs")

        filtered_scene_graphs = [
            sg for sg in scene_graphs
            if "id" not in sg or str(sg["id"]) not in self.processed_ids
        ]

        if not filtered_scene_graphs:
            logger.info("All scene graphs have already been processed")
            return []

        prompts_with_index = self._create_prompts(filtered_scene_graphs)
        results = await self._generate_prompts(prompts_with_index, self._validate_prompt)


        result_dict = {index: result for index, result in results}

        updated_scene_graphs = []
        for index, scene_graph in enumerate(filtered_scene_graphs):
            if index in result_dict:
                scene_graph = scene_graph.copy()
                result = result_dict[index]
                if isinstance(result, dict) and "prompt" in result:
                    result["prompt"] = self._remove_uuid_like(result["prompt"])
                scene_graph["prompt"] = result.get("prompt", "")
                updated_scene_graphs.append(scene_graph)
                if "id" in scene_graph:
                    self.processed_ids.add(str(scene_graph["id"]))


        if isinstance(self.existing_data, dict) and "difficulty" in self.existing_data:
            existing_items = self.existing_data["difficulty"]
            existing_ids = {str(item["id"]) for item in existing_items if "id" in item}
            updated_scene_graphs.extend(
                item for item in existing_items
                if "id" not in item or item["id"] not in self.processed_ids
            )
            self.existing_data["difficulty"] = updated_scene_graphs
        else:
            self.existing_data = {difficulty_str: updated_scene_graphs}

        self._save_data(self.existing_data)

        return updated_scene_graphs