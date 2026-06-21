import os
from pathlib import Path

import gradio as gr
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
LORA_FILE = Path(__file__).resolve().parent / "soy_diffusion.safetensors"

DEFAULT_PROMPT = (
    "feraljak, screaming, snail, pink_hair, hammer_and_sickle, tears, 4chan"
)

pipe = None


def _load_kohya_lora(pipe, lora_path: Path):
    """Load kohya SDXL LoRA via diffusers' SGM block mapper (UNet only)."""
    lora_result = StableDiffusionXLPipeline.lora_state_dict(
        str(lora_path),
        unet_config=pipe.unet.config,
    )
    if len(lora_result) == 3:
        state_dict, network_alphas, metadata = lora_result
    else:
        state_dict, network_alphas = lora_result
        metadata = None
    pipe.load_lora_into_unet(
        state_dict,
        network_alphas=network_alphas,
        unet=pipe.unet,
        adapter_name="soy_diffusion",
        metadata=metadata,
        _pipeline=pipe,
    )


def load_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    token = os.environ.get("HF_TOKEN")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        token=token,
    )
    pipe.vae = AutoencoderKL.from_pretrained(
        VAE_MODEL,
        torch_dtype=torch.float16,
        token=token,
    )
    if not LORA_FILE.is_file():
        raise FileNotFoundError(f"Missing LoRA weights: {LORA_FILE}")
    _load_kohya_lora(pipe, LORA_FILE)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(
    prompt,
    negative_prompt,
    lora_scale,
    steps,
    guidance,
    width,
    height,
    seed,
):
    p = load_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    image = p(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=generator,
        cross_attention_kwargs={"scale": float(lora_scale)},
    ).images[0]
    return image


with gr.Blocks(title="soy_diffusion") as demo:
    gr.Markdown(
        "# soy_diffusion\n"
        "SDXL LoRA demo. Prompt with booru-style tags — variant names like "
        "`feraljak`, `chudjak`, `cobson` steer the character/style."
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=3,
            )
            negative = gr.Textbox(
                label="Negative prompt",
                value="blurry, low quality, watermark, text",
                lines=2,
            )
            lora_scale = gr.Slider(0.0, 1.5, value=0.85, step=0.05, label="LoRA strength")
            steps = gr.Slider(10, 50, value=28, step=1, label="Steps")
            guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="CFG scale")
            with gr.Row():
                width = gr.Slider(512, 1024, value=1024, step=64, label="Width")
                height = gr.Slider(512, 1024, value=1024, step=64, label="Height")
            seed = gr.Slider(0, 2147483647, value=42, step=1, label="Seed")
            run = gr.Button("Generate", variant="primary")
        with gr.Column():
            out = gr.Image(label="Output")

    run.click(
        fn=generate,
        inputs=[prompt, negative, lora_scale, steps, guidance, width, height, seed],
        outputs=out,
    )

if __name__ == "__main__":
    demo.launch()
