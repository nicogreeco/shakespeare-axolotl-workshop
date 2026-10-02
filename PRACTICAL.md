# Give your LLM a character

Modern language models are rather good at making words follow other words. They are also rather large. Getting one onto a GPU, let alone teaching it something new, can turn into a memory puzzle. Every gigabyte we avoid using is also a small favour to the budget.

In this practical, you will try Nebius Serverless AI with a model from the [Qwen2.5 family](https://huggingface.co/collections/Qwen/qwen25). First, deploy a model behind an API and ask it a few questions from your DevLab. Then use a GPU Job to fine-tune it on a character's dialogue and serve the result through another endpoint. The dialogue comes from Shakespeare plays or public-domain character roleplay: Dracula, Sherlock Holmes, Odysseus, Dorian Gray, and the Cheshire Cat. This is an excuse to give a model a voice, and a soul  and to discover what the GPU actually has to hold while you do it.

You will use a **Job** for training, which finishes after running the task, and an **Endpoint** for inference, which stays available for requests. The DevLab is where you edit the training configuration and run the small [notebook](lab.ipynb). The containers already have the serving or training software installed. You do not have to build them.

## Every token counts

A language model takes the tokens it has seen so far and predicts the next one. A token can be a word, part of a word, or punctuation. Transformer-based models such as Qwen2.5 generate one token after another; their learned weights encode patterns that let them produce useful answers. The **B** in 7B means *billion* parameters. You can now guess why they are called *large* language models. The family includes [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [7B](https://huggingface.co/Qwen/Qwen2.5-7B), [14B](https://huggingface.co/Qwen/Qwen2.5-14B), and larger sizes.

A weight is a number stored as bits, the 0s and 1s in computer memory. Eight bits make one byte. There are different ways to represent numbers with those bits: **FP32** is a 32-bit floating-point format (4 bytes per weight), while **BF16** uses 16 bits (2 bytes). BF16 uses less memory at the cost of numerical precision; we do not need the details of its encoding here. For a first estimate, use:

```text
weight memory (bytes) ≈ number of parameters × bytes per parameter
weight memory (GB)    ≈ weight memory (bytes) / 1,000,000,000
```

Ignoring everything else, a 7B model needs roughly **28 GB in FP32 or 14 GB in BF16**. BF16 is widely used for LLM workloads because it halves weight storage compared with FP32 while usually retaining enough precision for training and inference. These are rounded model sizes and decimal GB; the exact checkpoint may differ.

The NVIDIA L40S in this exercise has [48 GB of GPU memory](https://www.nvidia.com/en-us/data-center/graphics-cards-for-virtualization/). The preset also lists 64 GiB of system RAM, which is a different pool. The GPU does the calculations and reads the model's weights again and again as it generates tokens. Its own high-bandwidth memory can supply those weights much faster than transferring them from system RAM, so in this setup the weights need to fit on the GPU. **Before creating an endpoint, estimate the BF16 weight memory for 14B and 32B. Which one looks plausible on one L40S?** Leave room beyond the weights.

Why the extra room? At each Transformer layer, a token produces attention **keys** and **values**. When the model generates the next token, it can reuse the earlier ones instead of recalculating the whole conversation. The stored keys and values are the **KV cache**. Longer conversations and more simultaneous requests need more cache; for a fixed model, its size grows roughly with the number of cached tokens. Our prompts are short, but the cache and other serving buffers still need GPU memory.

Training needs more. Besides weights, it keeps **activations**: intermediate results produced as examples pass through the layers. Some are needed again when the model calculates how to update its weights. Training also stores a **gradient** for each trainable weight and optimizer state. Adam optimizer, for example, tracks two running values per trainable weight. If we *pretend* all these values use BF16, full fine-tuning needs **2 bytes for a weight + 2 for its gradient + 2 + 2 for Adam = 8 bytes per parameter**, before activations. **Try this calculation for 7B. Does it fit in 48 GB?** Real optimizer states may use FP32, so the estimate can be optimistic. A larger microbatch processes more examples together: it usually needs more activation memory, while averaging more examples can make the gradient less noisy. [PyTorch's activation-memory guide](https://docs.pytorch.org/tutorials/beginner/mosaic_memory_profiling_tutorial.html) shows why keeping intermediate results matters.

**LoRA** takes a smaller route. Instead of changing a large weight matrix directly, it learns a correction represented by two much narrower matrices. Their width is controlled by the **rank** (`lora_r` in our YAML). The original model weights stay frozen; the learned matrices form an **adapter**. The full base model still has to fit, but gradients and optimizer states are needed only for the adapter. The saved adapter is small and must be loaded *with* that same base model to produce the new behaviour. [This LoRA explanation](https://huggingface.co/docs/peft/main/task_guides/lora_based_methods) goes further if you are curious.

## Put a model behind an API

Open the Nebius console and go to **Serverless AI → Endpoints → Create endpoint → Custom**. An endpoint starts a container, runs a command in it and exposes the port as an HTTPS API. Inside that container, [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) loads the model onto the GPU, manages requests and its KV cache, and returns generated text through an API compatible with the OpenAI Python SDK. Later it will also let us select the base model or its LoRA adapter by model name.

Use these values for the first endpoint. Give it a name you will recognize, such as `my-base-model`.

| Console field | Value |
|---|---|
| Image path | `docker.io/vllm/vllm-openai:latest` |
| Port | `8000` (HTTP) |
| Compute | GPU, regular NVIDIA L40S; 1 GPU, 16 vCPUs, 64 GiB RAM |
| Container disk | 100 GiB |
| Network / subnet | The available default for your project |
| Bearer-token authentication | Off for this classroom endpoint |
| Mounted volumes | None |

The image contains vLLM and its dependencies. Its entrypoint command tells it which model to download and how to run the API. The command below uses 7B as an example; replace the value after `--model` with the size you choose:

```bash
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.85 --host 0.0.0.0 --port 8000
```

`--dtype` selects BF16, `--max-model-len` limits the total input and output context, and `--gpu-memory-utilization` sets vLLM's GPU-memory budget. **Use your weight-memory estimate to choose a model before you launch it.** Could it fit with room for the KV cache and serving overhead? Which is the largest size you would try on this GPU, and what might stop it from working even if its weights fit? Change `--model`, create the endpoint and wait for it to become ready. Copy its HTTPS URL from **Network → Public endpoints**. The [Nebius endpoint guide](https://docs.nebius.com/serverless/tutorials/deploy-model) shows this console flow.

Open [the notebook](lab.ipynb), paste the URL, and play with its ready-made questions. Ask one of your own, too. In the endpoint's **Metrics** tab, how much GPU memory is used once the model is loaded? Is that close to your weight-only estimate? What happens to GPU utilization and memory while you send requests? Memory and GPU activity are different measures. vLLM can [reserve cache memory ahead of time](https://docs.vllm.ai/en/v0.10.1.1/configuration/optimization.html), so its memory graph may barely move for a short request even while the GPU works. [Nebius metrics may take a few minutes to appear](https://docs.nebius.com/serverless/monitoring). If you are curious, the **Logs** tab shows what the container and vLLM did while starting and serving.

Once you have played with it, stop this first endpoint; the final endpoint will let you query both the base and adapted model on one GPU.

## Give it a voice

Now for the training Job. We prepared chat-format JSONL datasets from [Shakespeare dialogue](https://huggingface.co/datasets/chaseharmon/6.7960_Shakespeare) and [public-domain character roleplay](https://huggingface.co/datasets/agentlans/practical-dreamer-RPGPT_PublicDomain). Each line is one conversation, for example:

```json
{"messages": [{"role": "user", "content": "I seem to have lost my way."}, {"role": "assistant", "content": "Perhaps the way has lost you."}]}
```

The example above is just an illustration. The character material is roleplay inspired by the characters, rather than quotations from the original books. Shakespeare consists of adjacent lines from plays; its language is Early Modern English. The user message supplies the context, and we train the model to predict the assistant response. In the YAML this is expressed by `roles_to_train: [assistant]` and `train_on_inputs: false`.

Choose a dataset. For a first run, the Cheshire Cat and other character datasets are short; Shakespeare is larger.

| Voice | Directory inside the Job |
|---|---|
| Cheshire Cat | `/inputs/datasets/public-domain/the-cheshire-cat/` |
| Dracula | `/inputs/datasets/public-domain/count-dracula/` |
| Sherlock Holmes | `/inputs/datasets/public-domain/sherlock-holmes/` |
| Odysseus | `/inputs/datasets/public-domain/odysseus/` |
| Dorian Gray | `/inputs/datasets/public-domain/dorian-gray/` |
| Shakespeare dialogue | `/inputs/datasets/shakespeare/` |

If the datasets are visible in your DevLab, open a `train.jsonl` file and look at a couple of lines. What are the input and the answer? Some prompts give away the character or setting; even an untrained base model might pick up that hint. That is fine for this exercise—we are here to try the workflow and see what it does.

### Make your training configuration

Open [training.yaml](training.yaml). This one YAML file is the starting point for every voice. [Axolotl](https://docs.axolotl.ai/) reads it to load the model and data, train LoRA, and choose checkpoints. Find `base_model`, `datasets`, `test_datasets`, `sequence_len`, `bf16`, and `lora_r`. **What is the longest training sequence this configuration allows? What precision will the job use? What LoRA parameters are used?**

Edit *both* dataset paths to your chosen directory, keeping `train.jsonl` for training and `validation.jsonl` for validation. Set `base_model` to the size you want to train: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-7B`, or `Qwen/Qwen2.5-14B`. If you want to compare with your first endpoint, use the same base model. Otherwise the final endpoint can still compare the adapted model with its own base model.

The default YAML is tuned for the Cheshire Cat at 7B. For the other characters, keep its training schedule. For Shakespeare, change `learning_rate` to `0.0002`, `max_steps` to `500`, `warmup_steps` to `10`, and `eval_steps` and `save_steps` to `125`. The longer dataset can support a longer run; we use a smaller learning rate and fewer steps for the short character datasets to limit overfitting.

Keep `lora_r: 16`, `micro_batch_size: 4`, and `gradient_accumulation_steps: 4` for a first 0.5B or 7B run. A **microbatch** of four means four examples are processed together. Axolotl accumulates gradients from four such microbatches before updating the adapter, so one update sees an **effective batch of 4 × 4 = 16 examples** on our single GPU. This gives a larger effective batch without holding all 16 examples' activations at once. Gradient checkpointing in the YAML saves more memory by recomputing some intermediate values.

**What about the training memory for your model?** Make one more estimate before launching. For rank 16, assume roughly **0.5% of the base parameters** are trainable LoRA parameters: if the base has `N` parameters, then `A ≈ 0.005 × N` adapter parameters. This is a rule of thumb from a 7B run, not a fixed property of LoRA; the percentage changes with model size, architecture and rank, and may be quite different for 0.5B. Write `P ≈ N` for frozen base parameters. Under the same simplified BF16 assumption as before:

```text
LoRA training state (bytes) ≈ 2P + 8A
                            ≈ 2N + 8 × (0.005N)
```

The `2P` is the frozen base model. The `8A` includes adapter weights, gradients and two Adam arrays. **Estimate this for 7B and 14B, then leave at least another 5 GB for activations, temporary buffers and the runtime. Which would you try on 48 GB?** This is a rough budget, not a promise: adapter states can use a different precision, and activations depend on sequence length and microbatch. [Axolotl's sizing guide](https://docs.axolotl.ai/docs/choosing_method.html) includes 13–14B LoRA runs on a single 48 GB GPU with short contexts and small microbatches.

If you try 14B, use `micro_batch_size: 2` and `gradient_accumulation_steps: 8` for the same effective batch of 16; this combination has completed a run with the present configuration. If your job runs out of GPU memory, try `1` and `16`. At a microbatch of one, the next options are a shorter `sequence_len` or a smaller model. A smaller microbatch can make the run slower. You can also try `lora_r: 8` and `lora_alpha: 16` as a separate experiment. Roughly how would halving rank affect `A` and the *total* memory estimate?

### Start a GPU Job

In the console, open **Serverless AI → Jobs → Create job**. Choose the **Axolotl template** and replace its defaults with these settings; the **Custom** route works too. The [Nebius fine-tuning tutorial](https://docs.nebius.com/serverless/tutorials/fine-tuning) has screenshots of the console.

| Console field | Value |
|---|---|
| Name | Something recognizable, such as `cheshire-training` |
| Image path | `docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1` |
| Compute, disk, network | L40S, 1 GPU / 16 vCPUs / 64 GiB RAM, 100 GiB disk, your project's default subnet |
| Timeout | 1 hour |
| First mounted volume | `workshop-input` at `/inputs`, read-only |
| Second mounted volume | `workshop-outputs` at `/outputs`, read-write |
| Files | Paste or upload your entire edited YAML at `/config/axolotl.yaml` |

A mounted Object Storage bucket appears as files in the container. For example, `s3://workshop-input/datasets/shakespeare/train.jsonl` becomes `/inputs/datasets/shakespeare/train.jsonl`. The input mount supplies datasets and the runner script; the output mount receives the adapter and run artifacts. In **Files**, paste the YAML from your editor, or download it from JupyterLab and upload it from your computer. See [Nebius's Job guide](https://docs.nebius.com/serverless/jobs/manage) for those controls.

Give the entrypoint this command. Choose a short group name and a unique run ID; use a new ID for each training attempt:

```bash
bash -c "bash /inputs/releases/v1/run_job.sh /config/axolotl.yaml my-group run-1"
```

This starts the included `run_job.sh` script with your YAML, group and run ID. Use lowercase letters, numbers and hyphens, with no spaces. It trains with Axolotl, then writes the results to `/outputs/my-group/runs/run-1/`. Choose a group name that identifies your team or experiment; the run ID distinguishes this attempt from others in that group. Use a new run ID for each attempt. Click **Create job** and watch the logs. The script prints the output path, and saves `training.log`, loss plots, and a small automatic comparison alongside `adapter/`.

While the Job runs, look for a line like this in its logs:

```text
trainable params: xxx || all params: xxx || trainable%: xxx
```

How many of *your* model's parameters are trainable? What percentage is that? Compare your earlier full-training estimate with `2P + 8A` using your log values (`P = all params − A`). The actual GPU memory can be higher because our estimate leaves out activations and runtime overhead. The Job's **Metrics** tab may show that difference. If you open `loss.csv` or `loss.svg` in your output, compare training and validation loss: can validation start rising while training loss keeps falling? That is one reason the job selects a validation checkpoint and may stop before `max_steps`.

## Put the character on stage

Once training has finished, browse **Storage → Object Storage → workshop-outputs → your group → runs → your run ID → adapter**. Check that `adapter_config.json` and `adapter_model.safetensors` are present. `adapter_config.json` also records the exact base model and LoRA rank.

Create another **Custom endpoint** with the same image, port, GPU, disk and network settings as before. Mount `workshop-outputs` at `/outputs` **read-only**. Use the exact base model and rank from `adapter_config.json`. Replace `my-group` and `run-1` below with the group name and run ID you chose:

```bash
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 16 --lora-modules character=/outputs/my-group/runs/run-1/adapter --host 0.0.0.0 --port 8000
```

The model name `character` refers to the adapter, whether you chose a character or Shakespeare. vLLM can [serve the base model and its LoRA adapter together](https://docs.vllm.ai/en/latest/features/lora/). Reconnect the notebook to this endpoint. Ask the base model and `character` the same question, then try your own prompts. Does it sound more like your chosen voice? Does it still answer the question? No need to score it: have a look and play.

When you are done, stop the endpoints in the console so they no longer reserve a GPU. Your training outputs remain in Object Storage.

## One more voice: Nebius Token Factory

Your LoRA adapter is one way to get a model to speak in a particular style. Now try a different route with **Nebius Token Factory**, a hosted API that gives you access to larger models without deploying them on your L40S. A model trained on broad collections of text may already know something about Shakespeare or familiar literary characters. A **system message** can ask it to adopt a voice; that is a prompt-time instruction, not fine-tuning. It may produce a convincing style, though it will not necessarily know every detail or stay in character.

Token Factory uses the same OpenAI-compatible chat format as vLLM. The client setup changes: use the Token Factory API address and an API key. The request still contains a model name and a list of messages. In that list, `system` gives the model its role and style, while `user` provides the question. See the [Token Factory quickstart](https://docs.tokenfactory.nebius.com/quickstart).

When you have the class API key, run the final Token Factory cells in `lab.ipynb`. The key is requested with hidden input and is not written into the cell. Select an available large model from the list shown by the notebook. Try either a character prompt, such as “You are Sherlock Holmes: answer with precise, observant reasoning and a restrained Victorian voice,” or a Shakespeare prompt, such as “Speak as a character in a Shakespeare play, using Early Modern English.” Then ask the same question you asked your fine-tuned model.

How close does the prompt-only answer get to the character or play style? What does the adapter add, if anything? You are comparing two ways to shape an answer: learned adapter weights and instructions in a system message. The results may vary by model and prompt.
