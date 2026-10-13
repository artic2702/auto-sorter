"""
Generate high-resolution application icons for Auto-Sorter (.ico and .png).
Creates multi-resolution ICO (256, 128, 64, 48, 32, 16) with clean modern design.
"""

from pathlib import Path
from PIL import Image, ImageDraw


def generate_app_icon(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle (deep blue / indigo gradient background)
    margin = 12
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=48,
        fill="#1e293b",  # slate-800
        outline="#3b82f6",  # blue-500
        width=4,
    )

    # Inner photo / camera lens circle
    center = size // 2
    lens_r = 72
    draw.ellipse(
        [center - lens_r, center - lens_r, center + lens_r, center + lens_r],
        fill="#0f172a",  # slate-900
        outline="#60a5fa",  # blue-400
        width=5,
    )

    # Face silhouette inside
    # Head circle
    head_r = 28
    head_y = center - 16
    draw.ellipse(
        [center - head_r, head_y - head_r, center + head_r, head_y + head_r],
        fill="#93c5fd",  # blue-300
    )

    # Shoulders arc/polygon
    shoulder_top = head_y + head_r + 6
    shoulder_bottom = center + 52
    draw.chord(
        [center - 50, shoulder_top - 10, center + 50, shoulder_bottom + 30],
        start=190,
        end=350,
        fill="#93c5fd",
    )

    # Biometric scanning brackets (corner focus marks)
    bracket_color = "#38bdf8"  # light cyan
    bw = 5
    blen = 28
    bpad = 38

    # Top-Left
    draw.line([(bpad, bpad), (bpad + blen, bpad)], fill=bracket_color, width=bw)
    draw.line([(bpad, bpad), (bpad, bpad + blen)], fill=bracket_color, width=bw)

    # Top-Right
    draw.line([(size - bpad, bpad), (size - bpad - blen, bpad)], fill=bracket_color, width=bw)
    draw.line([(size - bpad, bpad), (size - bpad, bpad + blen)], fill=bracket_color, width=bw)

    # Bottom-Left
    draw.line([(bpad, size - bpad), (bpad + blen, size - bpad)], fill=bracket_color, width=bw)
    draw.line([(bpad, size - bpad), (bpad, size - bpad - blen)], fill=bracket_color, width=bw)

    # Bottom-Right
    draw.line([(size - bpad, size - bpad), (size - bpad - blen, size - bpad)], fill=bracket_color, width=bw)
    draw.line([(size - bpad, size - bpad), (size - bpad, size - bpad - blen)], fill=bracket_color, width=bw)

    # Save PNG
    png_path = output_dir / "app.png"
    img.save(str(png_path), format="PNG")

    # Generate multi-size ICO
    ico_path = output_dir / "app.ico"
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    icon_images = [img.resize(s, Image.Resampling.LANCZOS) for s in sizes]
    icon_images[0].save(
        str(ico_path),
        format="ICO",
        sizes=sizes,
        append_images=icon_images[1:],
    )

    return ico_path, png_path


if __name__ == "__main__":
    out = Path(__file__).parent
    ico, png = generate_app_icon(out)
    print(f"Generated {ico} and {png}")
