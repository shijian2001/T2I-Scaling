import json
import logging
from collections import defaultdict
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
import sys
sys.path.append(str(root_dir))
logger = logging.getLogger(__name__)


def pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    elif word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    else:
        return word + "s"


def generate_qa_categories(scene_graph):
    qa = {
        "object": [],
        "count": [],
        "attribute": [],
        "relation": []
    }

    question_id = 1
    id_mapping = defaultdict(dict)

    processed_objects = set()

    for obj in scene_graph["objects"]:
        obj_name = obj["name"]

        if obj_name not in processed_objects:
            qa["object"].append({
                "question_id": question_id,
                "question": f"Is there the {obj_name} in the figure?",
                "answer": "yes",
                "dependencies": [0]
            })
            id_mapping["object"][obj_name] = question_id
            processed_objects.add(obj_name)
            question_id += 1

    obj_counter = defaultdict(int)
    for obj in scene_graph["objects"]:
        obj_counter[obj["name"]] += 1

    for name, count in obj_counter.items():
        base_id = id_mapping["object"].get(name, 0)
        if count == 1:
            qa["count"].append({
                "question_id": question_id,
                "question": f"Is there only 1 {name} in this figure?",
                "answer": "yes",
                "dependencies": [base_id] if base_id else [0]
            })
        else:
            qa["count"].append({
                "question_id": question_id,
                "question": f"Are there only {count} {pluralize(name, count)} in this figure?",
                "answer": "yes",
                "dependencies": [base_id] if base_id else [0]
            })
        question_id += 1

    for obj in scene_graph["objects"]:
        obj_name = obj["name"]
        base_id = id_mapping["object"].get(obj_name, 0)
        for attr in obj["attributes"]:
            for k, v in attr.items():
                qa["attribute"].append({
                    "question_id": question_id,
                    "question": f"Is the {obj_name}'s {k} {v}?",
                    "answer": "yes",
                    "dependencies": [base_id] if base_id else [0]
                })
                question_id += 1

    seen_relations = set()
    for rel in scene_graph["relations"]:
        key = (rel["subject"], rel["relation"], rel["object"])
        if key not in seen_relations:
            subj_id = id_mapping["object"].get(rel["subject"], 0)
            obj_id = id_mapping["object"].get(rel["object"], 0)

            dependencies = []
            if subj_id: dependencies.append(subj_id)
            if obj_id: dependencies.append(obj_id)
            if not dependencies: dependencies = [0]

            qa["relation"].append({
                "question_id": question_id,
                "question": f"Is the {rel['subject']} {rel['relation']} the {rel['object']}?",
                "answer": "yes",
                "dependencies": dependencies
            })
            question_id += 1
            seen_relations.add(key)

    return qa


def process_scene_graphs(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for difficulty in data.values():
        for scene in difficulty:
            scene_graph = scene["scene_graph"]
            scene["qa"] = generate_qa_categories(scene_graph)
            scene["valid"] = True

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
