# Give an LLM a character

A short practical on GPU memory, LoRA fine-tuning, and serving a model with Nebius Serverless AI. You can train on Shakespeare dialogue or public-domain character roleplay, then ask the base model and its adapter the same questions.

Start with [PRACTICAL.md](PRACTICAL.md). It walks through the Nebius console, asks you to estimate what fits on an L40S, and shows how to create the training Job and endpoints. Edit [training.yaml](training.yaml) for your chosen model and dataset, and use [lab.ipynb](lab.ipynb) to talk to the endpoints.

The optional command-line entry points and supporting scripts live in `src/`; they are not needed for the console-based practical. To prepare data locally, run `python3 src/prepare_data.py`. The older `axolotl.yaml` and `configs/` are separate run profiles, while [historical CLI notes](src/OPERATIONS.md) document earlier experiments.

The practical's Job executes the already published `/inputs/releases/v1/run_job.sh` from the `workshop-input` bucket. Moving files in this checkout does not update that Object Storage release.
