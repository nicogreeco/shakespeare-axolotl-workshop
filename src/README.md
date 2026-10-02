# Repository scripts

These files support the workshop and its earlier command-line workflow. The student practical uses the Nebius console; its training job runs the already published `workshop-input/releases/v1` bundle. Changes to the local runner do not change that bundle automatically.

- `prepare_data.py` downloads and converts the Shakespeare and public-domain character sources into chat-format JSONL files for Axolotl.
- `run_job.sh` is the training container entry point. It runs Axolotl, saves the adapter, collects the loss history, and makes a small base-versus-adapter comparison. The comparison and plotting Python files are kept beside it because the Object Storage release packages them together.
- `compare.py` loads a base model and its LoRA adapter, runs the same held-out prompts through both, and writes JSON and Markdown results.
- `plot_losses.py` turns Axolotl's trainer history into a CSV and an SVG plot of training and validation loss.
- `submit_job.sh` and `submit_endpoint.sh` are optional Nebius CLI launchers for creating training jobs and inference endpoints. The practical uses the console instead.
- `test_vllm_endpoint.ipynb` is an earlier, more detailed notebook for exploring a vLLM endpoint. The student-facing notebook is at the repository root.

The root `PRACTICAL.md`, `training.yaml`, and `lab.ipynb` are the student materials. `axolotl.yaml` and `configs/` contain separate profiles used by earlier runs.
