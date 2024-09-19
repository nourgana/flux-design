import torch
from diffusers import FluxPipeline
import PIL 


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "A Hublot Classic Fusion watch with a rose gold case and bezel. It features a white dial with rose gold hands and indices. The strap is black rubber."
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=768,
    num_inference_steps=20,
    generator=torch.manual_seed(42)
).images[0]
out.save("flux_pretrained.png")