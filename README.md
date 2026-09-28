# Shakespeare dialogue LoRA PoC

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` to answer one line of Shakespeare dialogue with the next line. The source dataset has no actor labels, so this trains a general play-character response, not a specific role.

## 1. Prepare the data on this VM

```bash
cd /mnt/filesystem-test/workshop-llm
python3 prepare_data.py
```

This downloads the pinned `shakespeare_unformatted.jsonl` from [chaseharmon/6.7960_Shakespeare](https://huggingface.co/datasets/chaseharmon/6.7960_Shakespeare), discards empty and duplicate `src`/`trg` pairs, and writes `data/train.jsonl` and `data/validation.jsonl`. The first 4,000 usable pairs train the model; the last 200 are held out, with a gap between them. Each row has only `user` and `assistant` messages; no system instruction is supplied. Only assistant tokens contribute to training loss.

## 2. Check and submit the job

The filesystem is in `eu-west1`. On this VM, profile `NicolaGreco` points to project `project-e01sbq2tpr00wr01f08fmk`; `NicolaNorth` was targeting `eu-north1` and caused a region mismatch. Select and check the project before submitting:

```bash
nebius profile activate NicolaGreco
nebius profile active
nebius config get parent-id
```

The launcher passes `project-e01sbq2tpr00wr01f08fmk` as `--parent-id` explicitly. To use another project in the same region, set `NEBIUS_PARENT_ID` to its ID. If you need a different CLI identity, set `NEBIUS_PROFILE` to its profile name; the default subnet is `vpcsubnet-e01ttsem3j1wq4hm5t` and can be overridden with `NEBIUS_SUBNET_ID`. The first cloud request under a profile may ask you to complete browser authentication.

```bash
./submit_job.sh --dry-run
./submit_job.sh
```

`--dry-run` is a local command preview: the Nebius CLI installed on this VM (0.12.235) does not provide a server-side `--dry-run` option. It checks that the prepared files exist and prints the exact create command without submitting a job.

The script runs this job configuration (with a fresh job name each time):

```bash
nebius ai job create \
  --name shakespeare-lora-YYYYMMDDHHMMSS \
  --parent-id project-e01sbq2tpr00wr01f08fmk \
  --subnet-id vpcsubnet-e01ttsem3j1wq4hm5t \
  --image docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1 \
  --platform gpu-h200-sxm --preset 1gpu-16vcpu-200gb \
  --disk-size 100Gi --timeout 2h \
  --volume computefilesystem-e01qd5gs6d17fnmtet:/workspace/data \
  --container-command bash \
  --args '-c "bash /workspace/data/workshop-llm/run_job.sh"'
```

The shared filesystem root is `/mnt/filesystem-test` on this VM and `/workspace/data` in the job. The job writes working files to its local disk, then copies the final LoRA adapter and a three-prompt base-versus-adapter comparison to `runs/<run-id>/` here.

## 3. Inspect results

Use the job name printed by `submit_job.sh`:

```bash
nebius ai job get-by-name --name <job-name>
JOB_ID=$(nebius ai job get-by-name --name <job-name> --format jsonpath='{.metadata.id}')
nebius ai job get "$JOB_ID"
nebius ai logs "$JOB_ID"
ls -lh runs/*/adapter/
cat runs/*/comparison.json
```

A completed run stores the adapter, `comparison.json`, `comparison.md`, `loss.csv`, and `loss.svg` under `/outputs/<group>/runs/<run-id>/`. The CSV contains `step`, `train_loss`, and `validation_loss`; the SVG plots both curves and marks the best validation-loss checkpoint. The comparison uses three held-out prompts without a system instruction for either model.

## 4. Retry comparison without training again

If training completed but the comparison script failed, run these commands **on the VM with the saved adapter**. The training job copies adapter files to the shared filesystem before starting comparison.

```bash
cd /mnt/filesystem-test/workshop-llm
git pull
ls -d runs/run-*/adapter
RUN_ID=run-YYYYMMDDTHHMMSSZ-NNNN  # replace with a directory name from the previous command
./submit_job.sh --compare-only "$RUN_ID" --dry-run
./submit_job.sh --compare-only "$RUN_ID"
```

This starts a new, inference-only Nebius job. It reads `runs/$RUN_ID/adapter` and writes `runs/$RUN_ID/comparison.json`; it does not repeat the 200 training steps. Use the new job name printed by the launcher with `nebius ai logs <job-id>` to check completion.

## 5. Public-domain character datasets

The second preprocessing option reads the downloaded
[`agentlans/practical-dreamer-RPGPT_PublicDomain`](https://huggingface.co/datasets/agentlans/practical-dreamer-RPGPT_PublicDomain)
file from `data/public_domain/train.jsonl.zst` and prepares all five characters:

```bash
python3 prepare_data.py --dataset public-domain
```

To rebuild only one character, use one of `count-dracula`, `sherlock-holmes`,
`the-cheshire-cat`, `odysseus`, or `dorian-gray`:

```bash
python3 prepare_data.py --dataset public-domain --character sherlock-holmes
```

Original conversations are shuffled with seed 42 and split 80/10/10 before any
examples are created. The original `system` message, including the character and
scenario description, is deliberately removed. Each example contains one or two
consecutive `user`/`assistant` exchanges, is limited to 500 whitespace-delimited
words, and ends with an assistant response. Windows do not overlap, so every
retained assistant response is a training target exactly once.

Replies containing only `*stage directions*` are excluded and start a new window;
replies containing both an action and spoken dialogue are retained. Without the
scenario, windows whose first user turn has fewer than six spoken words after
removing `*stage directions*` are also excluded as context-dependent. This drops
short continuations such as riddle answers while preserving their original text
in every retained example.

The generated files are kept separate by character:

```text
data/public_domain/processed/sherlock-holmes/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── summary.json
```

An abbreviated processed example looks like this:

```json
{
  "messages": [
    {"role": "user", "content": "Count Dracula, I thank you for your hospitality."},
    {"role": "assistant", "content": "Mr. Rivers, it is my pleasure to offer you shelter during this violent storm."},
    {"role": "user", "content": "Your reputation precedes you, sir."},
    {"role": "assistant", "content": "I am no stranger to the whispers and legends that surround my existence."}
  ]
}
```

Upload the desired directory to the input bucket:

```bash
aws s3 sync data/public_domain/processed/sherlock-holmes/ \
  s3://workshop-input/datasets/public-domain/sherlock-holmes/
```

Then select that character by changing only the dataset paths in `axolotl.yaml`:

```yaml
datasets:
  - path: /inputs/datasets/public-domain/sherlock-holmes/train.jsonl
    type: chat_template
    roles_to_train: [assistant]
test_datasets:
  - path: /inputs/datasets/public-domain/sherlock-holmes/validation.jsonl
    type: chat_template
    roles_to_train: [assistant]
```

The separate `test.jsonl` remains untouched during training and validation. With
the existing `roles_to_train: [assistant]` and `train_on_inputs: false` settings,
only the converted assistant messages contribute to the loss. The automatic
comparison sends the same held-out conversation prompt, without a character
instruction, to both the base model and the LoRA model.

## 6. Recommended training settings

Use separate step-based schedules for the two dataset families. The small
character datasets need far fewer steps than Shakespeare.

### Shakespeare

The 4,000-example Shakespeare training set can start with 500 steps:

```yaml
micro_batch_size: 4
gradient_accumulation_steps: 8  # effective batch size: 32
learning_rate: 0.0002
max_steps: 500
warmup_steps: 10

eval_strategy: steps
eval_steps: 125
save_strategy: steps
save_steps: 125
save_total_limit: 1
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
early_stopping_patience: 2
```

### Public-domain character

Each character has only about 160--250 training examples. Use a smaller learning
rate and effective batch size. Start with 120 steps and let validation-based early
stopping end the run if it stops improving:

```yaml
micro_batch_size: 4
gradient_accumulation_steps: 4  # effective batch size: 16
learning_rate: 0.00005
max_steps: 120
warmup_steps: 8

eval_strategy: steps
eval_steps: 15
save_strategy: steps
save_steps: 15
save_total_limit: 1
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
early_stopping_patience: 2
```

These profiles use integer step counts because this Axolotl version requires
`eval_steps` and `save_steps` with early stopping. Keep the evaluation and save
intervals equal so every evaluated checkpoint can become the best checkpoint.
Early stopping ends training after two consecutive epoch evaluations without an
improvement. The adapter copied to the run output is the checkpoint with the
lowest validation loss; validation perplexity gives the same ranking because it
is derived from that loss.

## References

- [Nebius Axolotl tutorial](https://docs.nebius.com/serverless/tutorials/fine-tuning) and [cookbook example](https://github.com/nebius/serverless-ai-cookbook/blob/main/training/axolotl-finetuning/README.md)
- [Axolotl chat dataset format](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)
- [Qwen2.5-1.5B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
