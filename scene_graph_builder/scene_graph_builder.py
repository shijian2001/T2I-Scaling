import json
from collections import defaultdict
import random
import networkx as nx
import argparse
import asyncio
import logging
from unittest.mock import AsyncMock
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
import sys

sys.path.append(str(root_dir))
from scene_graph_builder.sampler import SceneGraphSampler, SceneGraphDifficulty
from dataclasses import dataclass
import sys
import yaml
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
from api import StreamGenerator
from utils import JSONParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scene_graph_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RelationGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = """You are a scene graph relation generator. Given a set of objects with their attributes, generate plausible relations between them.
Follow these rules:
1. Only generate relations that make sense given the objects and their attributes
2. Each relation should be between two distinct objects
3. Only one relation allowed per object pair
4. Return only the specified number of relations"""
    max_concurrent_per_key: int = 300
    max_retries: int = 5


class RelationGenerator:
    def __init__(self, config: RelationGenerationConfig):
        self.config = config
        self.generator = StreamGenerator(
            model_name=self.config.model_name,
            api_keys=self.config.api_keys,
            max_concurrent_per_key=self.config.max_concurrent_per_key,
            max_retries=self.config.max_retries,
            rational=False
        )

    async def generate_relations(self, objects: List[Dict], num_relations: int) -> List[Dict]:
        if len(objects) != 2:
            raise ValueError("Must provide exactly 2 objects")
        if num_relations != 1:
            raise ValueError("Currently only supports generating 1 relation")

        prompt = self._create_relation_prompt(objects, num_relations)
        response = await self._generate_with_llm(prompt)

        if not response:
            raise ValueError("No valid relation from LLM inference")
        response = response.strip("*").strip()
        response = response.replace("\"", "").replace("'", "").strip()
        response = response.replace("*", "").replace("\n", "")

        relation = {
            "subject": objects[0]["name"],
            "object": objects[1]["name"],
            "relation": response
        }

        if "id" in objects[0]:
            relation["subject_id"] = objects[0]["id"]
        if "id" in objects[1]:
            relation["object_id"] = objects[1]["id"]

        return [relation]


    def _create_relation_prompt(self, objects: List[Dict], num_relations: int) -> str:
        subject = objects[0]
        object_ = objects[1]

        def format_attributes(attrs):
            if not attrs:
                return "no specific attributes"
            if isinstance(attrs, list):
                if all(isinstance(attr, dict) for attr in attrs):
                    attr_values = []
                    for attr in attrs:
                        attr_values.extend(attr.values())
                    return ', '.join(attr_values) if attr_values else "no specific attributes"
                else:
                    return ', '.join(str(attr) for attr in attrs)
            return str(attrs)

        subject_attrs = format_attributes(subject.get('attributes', []))
        object_attrs = format_attributes(object_.get('attributes', []))

        subject_info = f"- {subject['name']} (attributes: {subject_attrs})"
        object_info = f"- {object_['name']} (attributes: {object_attrs})"

        return f"""Please generate one visual relation between two objects in a text-to-image scene.
        Given these two objects:
subject: {subject_info}
object: {object_info}

## Requirements:
1. Generate a relation that makes semantic sense with "{subject['name']}" as the SUBJECT acting on "{object_['name']}" as the OBJECT
2. The relation must represent how the subject would naturally interact with or relate to the object

## Relation Categories (prioritize spatial relations):

### Spatial Relations (PREFERRED):
- **Position**: on, under, above, below, beside, next to, near, far from, in front of, behind
- **Containment**: inside, within, outside, around, surrounding, enclosing
- **Contact**: touching, against, leaning on, resting on, attached to, connected to
- **Orientation**: facing, pointing to, directed toward, aligned with
- **Relative Position**: left of, right of, between, among, across from

### Action Relations:
- **Physical Actions**: holding, carrying, pushing, pulling, lifting, dropping
- **Interaction**: using, operating, playing with, examining, touching
- **Movement**: approaching, moving toward, following, chasing

### Functional Relations:
- **Purpose**: for, used by, designed for, intended for
- **Ownership**: belongs to, owned by, part of
- **State**: connected to, linked to, associated with

## Examples:
- person + chair → "sitting on" or "standing beside"
- book + table → "on" or "lying on"
- car + road → "on" or "driving on"
- bird + tree → "perched on" or "flying near"
- cup + saucer → "on" or "resting on"
- dog + house → "inside" or "in front of"

## Output Format:
Return ONLY the base-form relation word/phrase - no subject or object names, no full sentences.
Examples: "on", "holding", "next to", "inside", "behind"
"""

    async def _generate_with_llm(self, prompt: str) -> str:
        try:
            results = self.generator.generate_stream_with_index(
                [(0, prompt)],
                self.config.system_prompt,
                self._validate_relation_response
            )
            async for index, result in results:
                if result is not None:
                    return result
            raise ValueError("No valid response from LLM")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _validate_relation_response(self, response: str) -> bool:
        return response.strip()


class SemanticEnhancer:
    def __init__(self, attributes_data: List[Dict]):
        self.attributes_data = attributes_data
        self.obj_to_attrs = {}

    def _decode_unicode_escapes(self, text: str) -> str:
        """Convert Unicode escape sequences to their corresponding characters."""
        try:
            return text.encode('utf-8').decode('unicode-escape')
        except:
            return text

    def _sample_unique_concept_attributes(self, all_attrs: List[Dict], num_attrs: int) -> List[Dict]:
        if not all_attrs or num_attrs <= 0:
            return []

        concept_groups = {}
        for attr in all_attrs:
            for concept, value in attr.items():
                if concept not in concept_groups:
                    concept_groups[concept] = []
                concept_groups[concept].append({concept: value})

        available_concepts = list(concept_groups.keys())
        selected_concepts = random.sample(
            available_concepts,
            min(num_attrs, len(available_concepts))
        )

        selected_attrs = []
        for concept in selected_concepts:
            selected_attr = random.choice(concept_groups[concept])
            selected_attrs.append(selected_attr)

        return selected_attrs

    def load_objects(self) -> List[Dict]:
        all_objects = []
        for category in self.attributes_data:
            for obj in category.get("objects", []):
                obj_name = obj["name"]
                self.obj_to_attrs[obj_name] = []
                for attr_type, attrs in obj.get("attributes", {}).items():

                    filtered_attrs = []
                    for attr in attrs:
                        if '(' in attr or ')' in attr:
                            continue
                        decoded_attr = self._decode_unicode_escapes(attr)
                        filtered_attrs.append({attr_type: decoded_attr})
                    self.obj_to_attrs[obj_name].extend(filtered_attrs)

                all_objects.append({
                    "name": obj_name,
                    "attributes": self.obj_to_attrs[obj_name]
                })

        return all_objects

    async def enhance_structure(
            self,
            G: nx.Graph,
            objects: List[Dict],
            relation_generator: 'RelationGenerator',
            duplicate_prob: float = 0.4
    ) -> Dict:
        object_nodes = [n for n in G.nodes if str(G.nodes[n]['type']) == 'object']
        relation_nodes = [n for n in G.nodes if str(G.nodes[n]['type']) == 'relation']

        objects = [
            obj for obj in objects
            if obj.get("attributes")
        ]

        name_to_attrs = {obj["name"]: obj["attributes"] for obj in objects}
        if len(objects) < len(object_nodes):
            raise ValueError(f"Required {len(object_nodes)} objects, but only {len(objects)} provided")

        relation_constraints = set()
        for rel_node in relation_nodes:
            connected = list(G.neighbors(rel_node))
            if len(connected) == 2:
                node_pair = tuple(sorted(connected))
                relation_constraints.add(node_pair)

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                scene_objects = []
                scene_relations = []
                node_to_obj = {}

                used_names = []
                name_counts = {}
                duplicate_targets = {}

                for node in object_nodes:
                    connected_nodes = []
                    for node_pair in relation_constraints:
                        if node in node_pair:
                            other_node = node_pair[0] if node_pair[1] == node else node_pair[1]
                            connected_nodes.append(other_node)

                    forbidden_names = set()
                    for connected_node in connected_nodes:
                        if connected_node in node_to_obj:
                            forbidden_names.add(node_to_obj[connected_node]["name"])

                    if used_names and random.random() < duplicate_prob:
                        available_duplicates = [
                            name for name in used_names
                            if (name_counts[name] < duplicate_targets.get(name, 2) and
                                name not in forbidden_names)
                        ]

                        if available_duplicates:
                            obj_name = random.choice(available_duplicates)
                        else:
                            available_objects = [
                                obj["name"] for obj in objects
                                if obj["name"] not in forbidden_names
                            ]
                            if available_objects:
                                obj_name = random.choice(available_objects)
                                if obj_name not in duplicate_targets:
                                    duplicate_targets[obj_name] = random.randint(2, 5)
                            else:
                                obj_name = random.choice(objects)["name"]
                                if obj_name not in duplicate_targets:
                                    duplicate_targets[obj_name] = random.randint(2, 5)
                    else:
                        available_objects = [
                            obj["name"] for obj in objects
                            if obj["name"] not in forbidden_names
                        ]
                        if available_objects:
                            obj_name = random.choice(available_objects)
                        else:
                            obj_name = random.choice(objects)["name"]

                        if obj_name not in duplicate_targets:
                            duplicate_targets[obj_name] = random.randint(2, 5)

                    used_names.append(obj_name)
                    name_counts[obj_name] = name_counts.get(obj_name, 0) + 1

                    connected_attrs = len([
                        n for n in list(G.neighbors(node))
                        if str(G.nodes[n]['type']) == 'attribute'
                    ])

                    if connected_attrs > 0 and obj_name in name_to_attrs:
                        assigned_attrs = self._sample_unique_concept_attributes(
                            name_to_attrs[obj_name],
                            connected_attrs
                        )
                    else:
                        assigned_attrs = []

                    node_to_obj[node] = {
                        "id": node,
                        "name": obj_name,
                        "attributes": assigned_attrs
                    }

                valid_assignment = True
                for rel_node in relation_nodes:
                    connected = list(G.neighbors(rel_node))
                    if len(connected) != 2:
                        continue

                    obj_node1, obj_node2 = connected
                    obj1, obj2 = node_to_obj[obj_node1], node_to_obj[obj_node2]

                    if obj1["name"] == obj2["name"]:
                        logger.warning(
                            f"Found relation between objects with same name: {obj1['name']} - {obj2['name']}")
                        valid_assignment = False
                        break

                if not valid_assignment:
                    logger.info(f"Attempt {attempt + 1}: Found relation between same-name objects, reassigning")
                    continue

                for rel_node in relation_nodes:
                    connected = list(G.neighbors(rel_node))
                    if len(connected) != 2:
                        continue

                    obj_node1, obj_node2 = connected
                    obj1, obj2 = node_to_obj[obj_node1], node_to_obj[obj_node2]

                    relations = await relation_generator.generate_relations([obj1, obj2], 1)
                    if relations:
                        relation = relations[0]
                        # Add object IDs to the relation
                        relation["subject_id"] = obj1["id"]
                        relation["object_id"] = obj2["id"]
                        scene_relations.append(relation)

                scene_objects = list(node_to_obj.values())

                duplicates = {name: count for name, count in name_counts.items() if count > 1}
                if duplicates:
                    logger.info(f"Duplicate objects in scene: {duplicates}")
                    for name, actual_count in duplicates.items():
                        target_count = duplicate_targets.get(name, 1)
                        logger.info(f"  {name}: target {target_count}, actual {actual_count}")

                return {
                    "objects": scene_objects,
                    "relations": scene_relations
                }

            except Exception as e:
                logger.warning(f"Semantic enhancement attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    raise ValueError(f"Failed to generate valid scene graph after {max_attempts} attempts")


def load_config(config_file: str) -> Dict[str, Any]:
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Config should be a YAML dictionary")
            return config
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid config file: {e}")


class SceneGraphBuilder:
    def __init__(self, attributes_file: str, config_file: str):
        with open(attributes_file) as f:
            self.attributes_data = json.load(f)

        self.config = load_config(config_file)

        self.difficulty_calc = SceneGraphDifficulty()
        self.structure_sampler = SceneGraphSampler(self.difficulty_calc)
        self.semantic_enhancer = SemanticEnhancer(self.attributes_data)

        keys_path = self.config.get("keys_path", "configs/keys.yaml")
        with open(keys_path, "r", encoding="utf-8") as f:
            api_keys = yaml.safe_load(f).get("keys", [])
        self.relation_generator = RelationGenerator(
            RelationGenerationConfig(
                model_name=self.config["model_name"],
                api_keys=api_keys,
                system_prompt=self.config.get("system_prompt", ""),
                max_concurrent_per_key=self.config.get("max_concurrent_per_key", 300),
                max_retries=self.config.get("max_retries", 5)
            )
        )

    async def build_batch(
            self,
            num_samples: int,
            min_diff: float,
            max_diff: float,
            max_retries: int = 3,
            duplicate_prob: float = 0.4  # 添加新参数
    ) -> Dict[str, List[Dict]]:
        results = []

        for sample_id in range(1, num_samples + 1):
            for retry in range(max_retries + 1):
                try:
                    G = self.structure_sampler.sample(d_min=min_diff, d_max=max_diff, max_iter=max_retries)

                    if not G:
                        raise ValueError("Invalid scene graph.")

                    actual_diff = self.difficulty_calc.calculate_difficulty(G)

                    if not (min_diff <= actual_diff <= max_diff):
                        raise ValueError(
                            f"Difficulty value {actual_diff:.2f} is out of range [{min_diff}, {max_diff}]"
                        )

                    all_objects = self.semantic_enhancer.load_objects()

                    scene = await self.semantic_enhancer.enhance_structure(
                        G, all_objects, self.relation_generator, duplicate_prob
                    )

                    results.append({
                        "id": sample_id,
                        "scene_graph": {
                            **scene,
                            "computed_diff": actual_diff
                        }
                    })
                    logger.info(f"Generating sample {sample_id}/{num_samples} (Difficulty: {actual_diff:.2f})")
                    break

                except Exception as e:
                    if retry == max_retries:
                        logger.warning(f"Sample generation failed: {str(e)}")
                    continue

        return {
            f"difficulty_{int(min_diff)}_{int(max_diff)}": results
        }


async def main():
    parser = argparse.ArgumentParser(description='Scene Graph Builder (Strictly Maintaining Graph Structure)')
    parser.add_argument("--attributes", required=True, help="Path to attributes JSON file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--min_diff", type=float, required=True, help="Minimum difficulty")
    parser.add_argument("--max_diff", type=float, required=True, help="Maximum difficulty")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--max_retry", type=int, default=3, help="Maximum retry attempts")
    parser.add_argument("--duplicate_prob", type=float, default=0.4,
                        help="Probability of object name duplication (0.0-1.0)")

    args = parser.parse_args()

    try:
        builder = SceneGraphBuilder(args.attributes, args.config)
        result = await builder.build_batch(
            args.num_samples,
            args.min_diff,
            args.max_diff,
            args.max_retry,
            args.duplicate_prob
        )

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

        success_count = len(result[next(iter(result))])
        print(f"\n✅ Completed! Successfully generated {success_count}/{args.num_samples} samples")
        print(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())