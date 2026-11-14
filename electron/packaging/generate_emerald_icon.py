"""
Generate crisp PalmPay-emerald multi-resolution Windows icon (`icon.ico` & `icon.png`)
for the Auto-Sorter Electron Desktop installer and taskbar shortcut.
"""

from pathlib import Path
from PIL import Image, ImageDraw


def generate_emerald_icon(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Render at 512x512 and downsample with Lanczos for razor-sharp anti-aliased edges
    canvas = 512
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded emerald square (#0e7c55) with subtle deep-green border (#0a5c3f)
    pad = 24
    draw.rounded_rectangle(
        [pad, pad, canvas - pad, canvas - pad],
        radius=108,
        fill="#0e7c55",
        outline="#0a5c3f",
        width=10,
    )

    # Inner crisp white photo card frame
    fx0, fy0, fx1, fy1 = 104, 122, 408, 390
    draw.rounded_rectangle(
        [fx0, fy0, fx1, fy1],
        radius=40,
        outline="#ffffff",
        width=24,
    )

    # Person silhouette / portrait badge on left of frame
    cx, cy = 202, 214
    draw.ellipse([cx - 32, cy - 32, cx + 32, cy + 32], fill="#ffffff")

    # Clean index lines on top-right
    draw.rounded_rectangle([274, 186, 358, 206], radius=10, fill="#b9ead6")
    draw.rounded_rectangle([274, 226, 336, 246], radius=10, fill="#b9ead6")

    # Landscape / album horizon line across bottom of photo card
    pts = [(126, 352), (206, 282), (268, 332), (324, 290), (386, 352)]
    draw.line(pts, fill="#ffffff", width=22, joint="curve")

    img_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
    png_path = out_dir / "icon.png"
    img_256.save(str(png_path), format="PNG")

    ico_path = out_dir / "icon.ico"
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    frames = [img.resize(s, Image.Resampling.LANCZOS) for s in sizes]
    frames[0].save(
        str(ico_path),
        format="ICO",
        sizes=sizes,
        append_images=frames[1:],
    )
    return ico_path, png_path


if __name__ == "__main__":
    here = Path(__file__).parent.resolve()
    ico, png = generate_emerald_icon(here)
    print(f"Generated {ico} and {png}")
