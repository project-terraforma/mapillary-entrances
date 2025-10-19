# src/pano_slices.py
from pathlib import Path
from PIL import Image
import math

def slice_equirectangular(
    img_path: Path,
    out_dir: Path,
    slice_deg: int = 60,
    overlap_deg: int = 0,
):
    """
    Split a 360° equirectangular image into equal angle slices.
    Returns a list of dicts with slice metadata (local angles in [0,360)).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    px_per_deg = W / 360.0
    step_deg = max(1, slice_deg - overlap_deg)
    step_px = max(1, int(round(step_deg * px_per_deg)))
    slice_px = int(round(slice_deg * px_per_deg))

    n = math.ceil((W + (slice_px - step_px)) / step_px)
    recs = []
    for k in range(n):
        x0 = (k * step_px) % W
        x1 = x0 + slice_px
        if x1 <= W:
            chip = im.crop((x0, 0, x1, H))
        else:
            right = im.crop((x0, 0, W, H))
            left = im.crop((0, 0, x1 - W, H))
            chip = Image.new("RGB", (slice_px, H))
            chip.paste(right, (0, 0))
            chip.paste(left, (right.size[0], 0))

        start_local = (x0 / W) * 360.0
        end_local   = (start_local + slice_deg) % 360.0
        center_local= (start_local + slice_deg/2) % 360.0

        fname = f"{img_path.stem}_slice{k:02d}_{int(round(start_local))}-{int(round(end_local))}.jpg"
        out_path = out_dir / fname
        chip.save(out_path, "JPEG", quality=92)

        recs.append({
            "slice_index": k,
            "path": str(out_path),
            "width_px": chip.size[0],
            "height_px": chip.size[1],
            "start_local_deg": start_local,
            "end_local_deg": end_local,
            "center_local_deg": center_local,
        })
    return recs
