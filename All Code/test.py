import os
import argparse
import cv2
from pathlib import Path

# ✅ Import your class (if it's in another file)
from ThresholdManager import ThresholdManager


def main():
    parser = argparse.ArgumentParser(
        description="Apply only Nuskhuri threshold methods and save output images."
    )

    parser.add_argument(
        "--image", "-i",
        default="/Users/sunnysideup/Desktop/untitled folder/samociqulo1709_page25_Nuskhuri.png",
        help="Path to input image"
    )

    parser.add_argument(
        "--out", "-o",
        default="/Users/sunnysideup/Desktop/untitled folder/nuskhuri_outputs",
        help="Directory to save processed images"
    )

    args = parser.parse_args()

    in_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Load image
    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    tm = ThresholdManager()

    try:
        img = tm.ensure_valid_image(img)
    except ValueError as e:
        raise SystemExit(f"Failed to load image: {e}")

    stem = in_path.stem

    # ✅ Run only Nuskhuri methods
    nuskhuri_results = tm.run_all_Nuskuri_Thresholds(img)

    saved_count = 0
    for name, variant in nuskhuri_results.items():
        out_path = out_dir / f"{stem}_{name}.png"
        cv2.imwrite(str(out_path), variant)
        saved_count += 1

    print(f"✅ Saved {saved_count} Nuskhuri thresholded images to:\n{out_dir.resolve()}")


if __name__ == "__main__":
    main()
