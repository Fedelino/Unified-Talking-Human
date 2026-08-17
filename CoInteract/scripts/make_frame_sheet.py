import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames found in {video_path}")
    frame_idx = max(0, min(int(frame_idx), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame), frame_idx, total


def main():
    parser = argparse.ArgumentParser(description="Create a simple video frame sheet.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", default="0,25,50,75")
    parser.add_argument("--thumb_width", type=int, default=360)
    args = parser.parse_args()

    video_path = Path(args.video)
    frame_ids = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    tiles = []
    total = None
    for idx in frame_ids:
        image, used_idx, total = read_frame(video_path, idx)
        scale = args.thumb_width / float(image.width)
        thumb = image.resize((args.thumb_width, int(round(image.height * scale))), Image.Resampling.LANCZOS)
        label_h = 28
        tile = Image.new("RGB", (thumb.width, thumb.height + label_h), (20, 20, 20))
        tile.paste(thumb, (0, label_h))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 6), f"frame {used_idx} / {total}", fill=(255, 255, 255), font=ImageFont.load_default())
        tiles.append(tile)

    sheet_w = max(t.width for t in tiles)
    sheet_h = sum(t.height for t in tiles)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=95)
    print(out)


if __name__ == "__main__":
    main()
