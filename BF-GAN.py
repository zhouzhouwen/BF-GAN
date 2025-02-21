import os
import re
from typing import List, Tuple, Union
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy

# zhouwen
# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


# ----------------------------------------------------------------------------

def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--label', help='Label tensor (e.g. \'0.212,0.505\')', type=parse_vec2, required=True, metavar='LABEL')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

def generate_images(
        network_pkl: str,
        seeds: List[int],
        label: Tuple[float, float],
        outdir: str,

):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)
    label_tensor = torch.tensor([label], device=device)

    for seed_idx, seed in enumerate(seeds):

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        img = G(z, label_tensor, truncation_psi=1.0, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        if img.shape[-1] == 3:
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        else:
            PIL.Image.fromarray(np.squeeze(img[0][:-2].cpu().numpy()), 'L').save(f'{outdir}/seed{seed:04d}.png')

        from PIL import Image

        def crop_and_save_image(image_path):
            img = Image.open(image_path)
            original_name = image_path.split('.')[0]
            start_x = 0
            width = 88
            move_steps = [8, 9, 10, 11, 8]

            for i, step in enumerate(move_steps):
                start_x += step
                crop_image = img.crop((start_x, 0, start_x + width, img.height))
                crop_image.save(f'{original_name}_image{i + 1}.png')
                start_x += width
                print('Generating image for seed %d - %d (%d/%d) ...' % (seed, i, seed_idx, len(seeds)))

            os.remove(image_path)

        crop_and_save_image(f'{outdir}/seed{seed:04d}.png')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
