# Shakespeare dialogue LoRA PoC

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` to answer one line of Shakespeare dialogue with the next line. The source dataset has no actor labels, so this trains a general play-character response, not a specific role.

## 1. Prepare the data on this VM

```bash
cd /mnt/filesystem-test/workshop-llm
python3 prepare_data.py
```

This downloads the pinned `shakespeare_unformatted.jsonl` from [chaseharmon/6.7960_Shakespeare](https://huggingface.co/datasets/chaseharmon/6.7960_Shakespeare), discards empty and duplicate `src`/`trg` pairs, and writes `data/train.jsonl` and `data/validation.jsonl`. The first 4,000 usable pairs train the model; the last 200 are held out, with a gap between them. Each row has `system`, `user`, and `assistant` messages. Only assistant tokens contribute to training loss.

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

Successful logs show training and validation loss through step 200. A completed run has `adapter_config.json`, adapter weights, and `comparison.json` in its `runs/<run-id>/` directory. The comparison uses three held-out prompts and the same system instruction for both models. Results may be modest after only 200 steps; this run first proves the pipeline works.

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

## References

- [Nebius Axolotl tutorial](https://docs.nebius.com/serverless/tutorials/fine-tuning) and [cookbook example](https://github.com/nebius/serverless-ai-cookbook/blob/main/training/axolotl-finetuning/README.md)
- [Axolotl chat dataset format](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)
- [Qwen2.5-1.5B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
