import os
from pathlib import Path

import gradio as gr
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
CHECKPOINT = Path(__file__).resolve().parent / "soy_diffusion.safetensors"

STYLE_TRIGGER = "soyjak"
DEFAULT_PROMPT = (
    "soyjak, feraljak, screaming, snail, pink_hair, hammer_and_sickle, tears, 4chan"
)
DEFAULT_NEGATIVE = (
    "photorealistic, photo, photograph, realistic, 3d, render, scenery, "
    "landscape, no humans, empty scene, painting, anime screenshot, "
    "watermark, text, blurry, low quality"
)

pipe = None


def load_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    token = os.environ.get("HF_TOKEN")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing fine-tuned SDXL checkpoint: {CHECKPOINT}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(CHECKPOINT),
        torch_dtype=torch.float16,
        use_safetensors=True,
        token=token,
    )
    pipe.vae = AutoencoderKL.from_pretrained(
        VAE_MODEL,
        torch_dtype=torch.float16,
        token=token,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def ensure_style_trigger(prompt: str) -> str:
    """Keep inference on the soyjak concept even if the user omits the trigger."""
    text = (prompt or "").strip()
    if not text:
        return STYLE_TRIGGER
    tokens = [t.strip().lower() for t in text.split(",")]
    if any(STYLE_TRIGGER in token for token in tokens):
        return text
    return f"{STYLE_TRIGGER}, {text}"


def generate(
    prompt,
    negative_prompt,
    steps,
    guidance,
    width,
    height,
    seed,
):
    p = load_pipeline()
    prompt = ensure_style_trigger(prompt)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    image = p(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=generator,
    ).images[0]
    return image


with gr.Blocks(title="soy_diffusion") as demo:
    gr.Markdown(
        "# soy_diffusion\n"
        "Fine-tuned SDXL (not a LoRA). **Every image should contain a soyjak.** "
        "Extra objects and settings in the prompt are fine — they appear *with* "
        "the soyjak, not instead of it. Start with `soyjak`, then a variant and "
        "any other tags. The trigger is prepended automatically if you leave it out."
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
                value=DEFAULT_NEGATIVE,
                lines=2,
            )
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
        inputs=[prompt, negative, steps, guidance, width, height, seed],
        outputs=out,
    )

if __name__ == "__main__":
    demo.launch()
