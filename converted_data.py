import sys, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def clean_image(img):
    """
    Refined Cleaner:
    1. Loop (Interpolation) -> Prevents Black Lines.
    2. Std Dev Check -> Prevents deleting white shirts.
    """
    cleaned = img.copy()
    h, w = cleaned.shape
    
    # 1. Fix Rows (Horizontal)
    for y in range(h):
        # Check: Is it Bright (>240) AND Flat (std < 10)?
        if np.mean(cleaned[y, :]) > 240 and np.std(cleaned[y, :]) < 10:
            # Fix: Copy row from above (Interpolate)
            cleaned[y, :] = cleaned[y-1, :] if y > 0 else 0

    # 2. Fix Cols (Vertical)
    for x in range(w):
        # Check: Is it Bright (>240) AND Flat (std < 10)?
        if np.mean(cleaned[:, x]) > 240 and np.std(cleaned[:, x]) < 10:
            # Fix: Copy col from left (Interpolate)
            cleaned[:, x] = cleaned[:, x-1] if x > 0 else 0
            
    return cleaned

def show_comparison(original, cleaned, label):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(original, cmap='gray'); plt.title(f"Bad: {label}")
    plt.subplot(1, 2, 2); plt.imshow(cleaned, cmap='gray'); plt.title("Fixed")
    plt.tight_layout(); plt.show()

def main():
    if len(sys.argv) < 7: sys.exit("Usage: python clean_data.py <dir> h w c <npz> <fix>")
    
    img_dir, h, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out_file, do_fix = sys.argv[5], int(sys.argv[6])

    print(f"Scanning {img_dir}...")
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    images, labels, shown = [], [], 0

    for fname in files:
        path = os.path.join(img_dir, fname)
        try:
            label = int(fname.split("-")[0])
            img = np.asarray(Image.open(path).convert("L"))
            if img.shape != (h, w): continue

            if do_fix:
                original = img.copy()
                img = clean_image(img)
                # Show first 3 fixes
                if shown < 10 and not np.array_equal(original, img):
                    print(f"Visualizing fix for {fname}...")
                    show_comparison(original, img, label)
                    shown += 1

            images.append(img)
            labels.append(label)
        except: continue

    np.savez(out_file, images=np.array(images), labels=np.array(labels))
    print(f"Done! Saved {len(images)} images to {out_file}")

if __name__ == "__main__":
    main()
