import os
import time
import argparse
from dataclasses import dataclass

import torch
import numpy as np
import tqdm
from PIL import Image

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, load_ae, load_clip, load_flow_model, load_t5)

@dataclass
class SamplingOptions:
    source_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    """
    Extract attention features from images using FLUX model inversion.
    
    This function processes images (single file or directory) and extracts
    K and V attention features at each layer and timestep, storing them to disk
    for later use in computing aging directions.

    Args:
        args: Command-line arguments containing:
            - source_img_dir: Path to image file or directory
            - source_prompt: Description of source image (or generated from age/gender/ethnicity)
            - feature_path: Output directory for features
            - age, gender, ethnicity: Parameters for auto-generating prompts
            - num_steps: Number of inversion steps
            - save_feature: Flag to enable feature extraction
        device: PyTorch device (cuda/cpu)
        offload: Enable CPU offloading for memory efficiency
    """
    torch.set_grad_enabled(False)
    name = args.name
    source_prompt = args.source_prompt
    guidance = args.guidance
    num_steps = args.num_steps
    offload = args.offload

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)
    
    img_list = make_dataset(args.source_img_dir) if os.path.isdir(args.source_img_dir) else [args.source_img_dir] # if the source_img_dir is a single image, make it a list
    for source_img_dir in tqdm.tqdm(img_list, desc="Processing images"):
        init_image = None
        init_image = np.array(Image.open(source_img_dir).convert('RGB'))
        
        shape = init_image.shape

        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

        init_image = init_image[:new_h, :new_w, :]

        width, height = init_image.shape[0], init_image.shape[1]
        init_image = encode(init_image, torch_device, ae)

        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            source_prompt=source_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = rng.seed()
        # print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        # t0 = time.perf_counter()

        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)

        info = {}
        source_img_name = os.path.basename(source_img_dir).split('.')[0]
        
        # Extract age from filename (assumes format like "30_0.jpg" where 30 is the age)
        try:
            image_age = int(source_img_name.split('_')[0])
        except (ValueError, IndexError):
            print(f"Warning: Could not extract age from filename '{source_img_name}'. Expected format: 'age_xxx.jpg' (e.g., '30_0.jpg')")
            image_age = None
        
        # Apply age-based filtering
        if image_age is not None and (args.age_filter_young and args.age_filter_old) and args.gender:
            ethnicity_str = f'{args.ethnicity} ' if args.ethnicity else ''
            if args.age_filter_young[0] <= image_age <= args.age_filter_young[1] or args.age_filter_old[0] <= image_age <= args.age_filter_old[1]:
                source_prompt = f'A photo of {ethnicity_str}{args.gender} at {image_age} years old'
            else:
                print(f"Skipping {source_img_name} because the age is not in the range")
                continue
        
        
        info['feature_path'] = os.path.join(args.feature_path, source_img_name)
        if args.save_feature:
            os.makedirs(info['feature_path'], exist_ok=True)
        info['feature'] = {}
        info['feature_buffer_V'] = {}
        info['feature_buffer_K'] = {}
        info['inject_step'] = args.inject
        info['save_feature'] = args.save_feature
        inp = prepare(t5, clip, init_image, prompt=source_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # inversion initial noise
        z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        if args.save_feature:
            t_0 = time.perf_counter()
            print("starting saving features")
            torch.save(info['feature_buffer_V'], os.path.join(info['feature_path'], 'features_V.pth'))
            torch.save(info['feature_buffer_K'], os.path.join(info['feature_path'], 'features_K.pth'))
            print(f"saving feature finished for {source_img_name} in {time.perf_counter() - t_0:.1f}s")




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_prompt', type=str, default='',
                        help='Describe the content of the source image. If not provided, will be generated from image filename (age extracted from name like "30_0.jpg"), --gender, and --ethnicity')
    parser.add_argument('--gender', type=str, default=None, choices=['male', 'female'],
                        help='Gender of the person in the image (used to generate source_prompt if not provided). Required for auto-generated prompts.')
    parser.add_argument('--ethnicity', type=str, default=None,
                        help='Ethnicity of the person in the image (e.g., white, black, asian, hispanic). Optional.')
    parser.add_argument('--age_filter_young', type=int, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Age range for young group filtering (e.g., --age_filter_young 25 35). If specified, only process images with age in this range.')
    parser.add_argument('--age_filter_old', type=int, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Age range for old group filtering (e.g., --age_filter_old 65 75). If specified, only process images with age in this range.')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='The path to save the extracted features')
    parser.add_argument('--guidance', type=float, default=2,
                        help='Guidance scale')
    parser.add_argument('--num_steps', type=int, default=15,
                        help='The number of timesteps for inversion')
    parser.add_argument('--inject', type=int, default=3,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--offload', action='store_true', 
                        help='Enable CPU offloading to reduce GPU memory usage')
    parser.add_argument('--save_feature', action='store_true', default=True, 
                        help='Enable feature extraction and saving')

    args = parser.parse_args()

    main(args)
