import torch
from diffusers import FluxPipeline
import PIL 

lora_model = "output/models/checkpoint-1600"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.load_lora_weights(lora_model)

prompt= "A Hublot watch featuring a rose gold case and a bezel set with diamonds. It has a black dial with rose gold hands and indices, complemented by a black rubber strap."
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=768,
    num_inference_steps=20,
    generator=torch.manual_seed(0)
).images[0]
out.save("flux_custom.png")