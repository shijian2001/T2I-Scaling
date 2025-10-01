# T2I-Scaling
We leverage scene graphs and introduce a novel difficulty criterion along with a corresponding adaptive Markov Chain Monte Carlo (MCMC) graph sampling algorithm. Using this difficulty-aware approach, we generate training datasets for Group Relative Policy Optimization (GRPO) comprising prompts and question-answer pairs with varying complexity levels.

## Installation

```shell
pip install -r requirements.txt
```

## Data Generation Pipeline

### Step 1: Graph Asset Generation
Configure parameters in the respective YAML files and run:

```shell
python scripts/generate_objects.py configs/object_gen.yaml
python scripts/generate_attributes.py configs/attribute_gen.yaml  
python scripts/generate_relations.py configs/relation_gen.yaml
```

### Step 2: Graph Sampling via MCMC
Configure sampling parameters and execute:

```shell
python scripts/sample.py configs/sample.yaml
```

### Step 3: Prompt and QA Generation
Generate training prompts and question-answer pairs for reward calculation in GRPO:

```shell
python scripts/generate_prompts.py configs/prompts.yaml
python scripts/generate_qa.py configs/qa.yaml
```

## Training

For training implementation, refer to the open-source project:
[https://github.com/XueZeyue/DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

Integrate our provided `train/curr_sampler.py` into your training script to enable curriculum learning capabilities.

## Evaluation

Refer to the following evaluation frameworks:

- **GenEval**: [https://github.com/djghosh13/geneval](https://github.com/djghosh13/geneval)
- **DPG**: [https://github.com/TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA)  
- **TIFA**: [https://github.com/Yushi-Hu/tifa](https://github.com/Yushi-Hu/tifa)
- **DSG**: [https://github.com/j-min/DSG](https://github.com/j-min/DSG)
- **T2I-CompBench**: [https://github.com/Karine-Huang/T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench)