# storyboard-to-video
# Storyboard-to-Video Pipeline

An end-to-end pipeline that turns structured scene descriptions into a stitched short film using pre-trained video diffusion models — built on free-tier T4 GPU hardware.

> Built as a practical demo for WAN 2.1 (Alibaba), presented in a Foundation Models course.

---

## What it does

1. Scene prompts define each shot with cinematic detail — subject, camera motion, lighting, depth of field
2. Each prompt is passed through a video diffusion model (CogVideoX-2b) to generate a ~2s clip
3. MoviePy stitches all clips into a single final MP4 on CPU — no GPU needed for this step

---

## Pipeline

```
Scene Prompts → T5 Text Encoding → 3D Diffusion Loop (25 steps) → VAE Decode → MoviePy Stitch
```

- **Model:** CogVideoX-2b — same HuggingFace Diffusers API as WAN 2.1 (one-line swap)
- **Text encoder:** T5 — conditions the denoising at every step via cross-attention
- **Diffusion:** 3D Diffusion Transformer, attends across space + time jointly
- **Stitching:** MoviePy `concatenate_videoclips()` — runs on CPU

---

## Hardware & Memory Optimizations

Designed to run on a **free Colab T4 (15GB VRAM)**. Four optimizations applied together:

| Optimization | Why |
|---|---|
| `torch.float16` | Halves VRAM vs float32 |
| `enable_sequential_cpu_offload()` | Moves layers CPU↔GPU one at a time |
| `vae.enable_tiling()` | Splits video into spatial tiles during VAE decode |
| `vae.enable_slicing()` | Processes frames in chunks to prevent OOM |

Without all four, the model hits OOM immediately on a T4.

---

## Results

| Scene | Prompt | Output |
|---|---|---|
| The Field | Red poppies and wildflowers, ground-level dolly forward, cinematic 4K | Generated correctly |
| The Sunflower | Row of sunflowers swaying, slow pan, warm afternoon light | Generated correctly |
| The Butterfly | Monarch butterfly on lavender, extreme close-up | Failed — flat yellow frame (model scale limitation) |

The butterfly clip is a documented failure mode: fine-grained subjects consistently degrade to flat color washes at 2B parameter scale.

---

## Generation Parameters

```python
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

output = pipe(
    prompt=scene["prompt"],
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=25,
    num_frames=17,
    guidance_scale=6.0,
    height=480,
    width=720,
).frames[0]
```

---

## How to Run

1. Open `storyboard_to_video.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4)
3. Run all cells in order
4. Generated clips and final stitched video download automatically

---

## Upgrading to WAN 2.1

The pipeline is model-agnostic. To use WAN 2.1 14B on an A100:

```python
# Change these two lines:
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16)

# And update settings:
NUM_FRAMES = 49
FPS = 16
WIDTH = 832
```

Everything else stays the same.

---

## Future Scope

- **User-defined scene templates** — structured input form (subject, camera, lighting, mood) instead of hardcoded prompts
- **LLM scene decomposition** — Gemini/GPT-4o generates scene prompts from a one-line story idea automatically *(integration built, hit free-tier quota limits)*
- **WAN 2.1 14B on A100** — one-line model swap for 720P output and stronger prompt adherence
- **Cross-scene consistency** — shared style conditioning to keep lighting and color grade consistent across clips

---

## Tech Stack

- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
- [MoviePy](https://github.com/Zulko/moviepy)
- PyTorch
