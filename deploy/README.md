# Strata on GPUStack

Strata runs **Qwen3.8-Flash-Next** (a 125-billion-parameter mixture-of-experts model) on one
consumer NVIDIA card plus system RAM. This folder turns it into a **GPUStack custom inference
backend**, so you can deploy and manage it from GPUStack's web UI.

- The inference engine, the Python server and the web app are all inside one Docker image:
  `ghcr.io/zhangyi1976/niko1221-strata:v0.1.40`.
- It speaks the **OpenAI** and **Anthropic** HTTP APIs, and it has its own web chat and a live
  monitor of the model and the GPU.
- The image is built automatically by GitHub Actions (`.github/workflows/build-push.yml`) and
  pushed to GHCR. You do not build it yourself.

## What the host needs

- One **NVIDIA** graphics card with **12 GB or more of video memory** (16-24 GB is comfortable).
- **nvidia-container-toolkit** installed on the host, so Docker can see the card.
- **About 100 GB of free disk** for the data volume (the two model shards are ~68 GB, plus the
  ~5 GB MTP draft layer and the prepared pack).
- Enough **system RAM** to hold the model's experts while they are not on the card. The entrypoint
  sets the low-RAM mode automatically; a memory-capped container asks for it itself.

## 1. Import the backend

1. Open GPUStack and go to **Inference Backends**.
2. Click **Add Backend**, choose **Custom**, and import this folder's
   [`backend.yaml`](backend.yaml).
3. You now have a backend named **strata**, version **0.1.40**, with the health check on
   `/health`.

`backend.yaml` sets the container's entrypoint to `/opt/strata/deploy-entrypoint.sh` and its
arguments to `{{model_path}} {{port}}`. GPUStack substitutes those two placeholders when it
schedules the container: the folder it downloaded the model files into, and the port it gave the
instance. The entrypoint turns those into a Strata setup and a server start.

## 2. Deploy the model

1. Open the **Deployments** page and add a model.
2. Name it, for example `Qwen3.8-Flash-Next-IQ2_XS`.
3. For the **backend**, choose **strata**.
4. Add the model files. Strata's Qwen3.8-Flash-Next is two GGUF shards; add **both**:

   ```
   hf.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
   hf.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
   ```

   (Other quants in the same repo work the same way: `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`.)
5. Attach the model to a worker that has the card, and start it.

GPUStack downloads the two shards into its own model folder and passes that folder to the
container. The entrypoint sees the `.gguf` files there and tells `setup.py` to read them with
`--gguf-dir`, so it only has to fetch the ~5 GB MTP draft and the prepared pack, then it loads
the model and opens the API port.

If the model files are not there (for example you deployed the model without listing the files),
the entrypoint waits up to 10 minutes for GPUStack's download, and if they never appear it lets
Strata download the model itself - the same way the project's own Docker image works. Either way
the container ends up serving the model.

A ready-made starting point for the deployment is [`model-iq2-xs.yaml`](model-iq2-xs.yaml).

## 3. Wait for it to come up

The first start takes a few minutes: it fetches the MTP draft and the prepared pack, then loads
the ~68 GB model. While it is doing that the instance stays **STARTING** and `/health` does not
answer yet - that is normal, not an error. When the model is loaded the API port opens,
`/health` answers, and the instance becomes **RUNNING**. You can then talk to it in GPUStack's
playground or from any app that speaks the OpenAI or Anthropic API.

## Settings (environment variables)

Every setting is the same environment variable the project's own image uses. Set them on the
backend version's `env` (in `backend.yaml`) or on the deployment, whichever you prefer. The
defaults in `backend.yaml` run the model at its default size with a 32,000-token context.

| Variable     | Meaning                                                                                             | Default   |
| ------------ | --------------------------------------------------------------------------------------------------- | --------- |
| `FAMILY`     | Model family: `qwen`, `swift`, `coder`, `unsloth`                                                    | `qwen`    |
| `MODEL`      | Quant/size: `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`                                                     | `IQ2_XS`  |
| `CONTEXT`    | Context length in tokens                                                                              | `32768`   |
| `VISION`     | `no`, `yes` (pictures on the card) or `cpu` (pictures on the processor)                              | `no`      |
| `KV`         | KV-cache quant: `int8`, `q4_0`, `k8v4`; empty leaves setup's own default                             | *(empty)* |
| `LOW_RAM`    | `auto`, `on` or `off`: with `on` the experts come from the pack, not from RAM                        | `auto`    |
| `GPUS`       | Several cards, for example `0,1` or `all`: one model across them                                      | *(empty)* |
| `GPU`        | One card, numbered as `nvidia-smi` numbers them                                                       | *(empty)* |
| `LAYER_SPLIT`| With `GPUS`: where each later card's layers start (default: automatic)                               | *(empty)* |
| `API_KEY`    | Required if the server is reachable beyond the worker; see below                                      | *(empty)* |
| `REINSTALL`  | Set to `1` to run the setup pass again after a model is already set up                               | `0`       |

To change a setting for a model that is already set up (context, vision, KV, host, API key), set
`REINSTALL=1` once; the setup pass re-runs and the new config is recorded. Switching between
models that are already on the volume needs no setup pass.

### Running on more than one card

Custom backends run on a single worker, but that worker can have several cards. Set `GPUS` to the
cards to use, for example `GPUS=0,1`, and optionally `LAYER_SPLIT` to control where the later
cards' layers start. The `{{gpu_count}}` and `{{gpu_ids}}` placeholders in `run_command` reflect
the cards assigned on that worker if you want to use them.

### The API key rule

Strata answers `/health` before the API-key check, so the health check works without a key. But if
the server is reachable beyond the worker's loopback (and a GPUStack deployment exposes it on the
worker's IP), set `API_KEY` to a secret; otherwise anyone who can reach the port can use the
model. Any key value works with the API clients; the model name is ignored.

## Disk layout

Mount a volume at `/data` (the image declares it). That is where the model files, the prepared
pack, the MTP draft, the config and the logs live, so a worker reboot does not lose them. Point
GPUStack's model storage at a folder on that volume if you want GPUStack's own download to reuse
the files Strata already has.

## Files in this folder

| File                     | What it is                                                        |
| ------------------------ | ----------------------------------------------------------------- |
| [`backend.yaml`](backend.yaml)     | The custom backend: image, entrypoint, run command, health check, defaults. |
| [`model-iq2-xs.yaml`](model-iq2-xs.yaml) | A starting-point model deployment for the default IQ2_XS quant.    |
| [`README.md`](README.md)           | This guide.                                                       |

The image itself is built from the repository's [`Dockerfile.deploy`](../Dockerfile.deploy)
by the [`build-push.yml`](../.github/workflows/build-push.yml) workflow and tagged
`ghcr.io/zhangyi1976/niko1221-strata:v0.1.40`.
