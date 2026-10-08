"""Render the existing paper pipeline PDF as a compact web image."""

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image

site = Path(__file__).resolve().parents[1]
source = site.parent / "FwdSlam/paper/iclr2027/figs/pipeline_used_recolored.pdf"
page = fitz.open(source)[0]
image = Image.open(BytesIO(page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png"))).convert("RGB")
target = site / "assets/pipeline-figure.webp"
image.save(target, "WEBP", quality=92, method=6)
print(f"{target.relative_to(site)}: {image.width} × {image.height}")
