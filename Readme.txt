================================================================================
LINE-BY-LINE EXPLANATION OF clean_data.py
================================================================================

--------------------------------------------------------------------------------
BLOCK 1: IMPORTS (THE TOOLBOX)
--------------------------------------------------------------------------------

1.  import sys, os
    - We import 'sys' to access command-line arguments (the text you type after 'python').
    - We import 'os' to interact with the operating system, like reading folder contents.

2.  import numpy as np
    - We import the Numpy library and nickname it 'np'. 
    - This is our math engine. It handles the images as matrices (grids of numbers).

3.  from PIL import Image
    - We import the 'Image' tool from the Pillow library.
    - We need this to open the actual .png files from your hard drive.

4.  import matplotlib.pyplot as plt
    - We import the plotting library and nickname it 'plt'.
    - We need this to create the popup windows that show the "Before vs. After" images.

--------------------------------------------------------------------------------
BLOCK 2: THE CLEANING FUNCTION (The Logic)
--------------------------------------------------------------------------------

5.  def clean_image(img):
    - We define a new function named 'clean_image'. 
    - It accepts one input variable 'img' (the image matrix).

6.      cleaned = img.copy()
    - We create a clone of the image and call it 'cleaned'. 
    - Crucial Step: We never edit the original 'img' directly while reading it.

7.      h, w = cleaned.shape
    - We ask for the dimensions of the matrix. 
    - 'h' gets the height (28 rows), 'w' gets the width (28 columns).

8.      # 1. Fix Rows (Horizontal)
    - Comment: We are starting the pass to fix horizontal lines.

9.      for y in range(h):
    - We start a loop that counts from 0 up to 27 (the height).
    - 'y' represents the current row number we are checking.

10.         if np.mean(cleaned[y, :]) > 240 and np.std(cleaned[y, :]) < 10:
    - The "Smart" Condition. We check two things about the current row (cleaned[y, :]):
      1. Is the average brightness > 240? (Is it a white line?)
      2. Is the Standard Deviation < 10? (Is it perfectly flat/smooth?)
    - If BOTH are true, we identify this row as Corruption. 
    - This logic saves textured items like white shirts (which have high Std Dev).

11.             cleaned[y, :] = cleaned[y-1, :] if y > 0 else 0
    - The Fix (Interpolation). 
    - We take the row ABOVE the current one (y-1) and copy it over the bad row.
    - "if y > 0 else 0": This is a safety check. If the very first row (0) is bad, 
      we can't look above it, so we just set it to 0 (black).

12.     # 2. Fix Cols (Vertical)
    - Comment: Now we start the second pass for vertical lines.

13.     for x in range(w):
    - We start a loop that counts from 0 up to 27 (the width).
    - 'x' represents the current column number.

14.         if np.mean(cleaned[:, x]) > 240 and np.std(cleaned[:, x]) < 10:
    - We run the exact same check as line 10, but for columns (cleaned[:, x]).
    - We look for bright, flat vertical lines.

15.             cleaned[:, x] = cleaned[:, x-1] if x > 0 else 0
    - The Fix. We copy the column to the LEFT (x-1) over the bad column.
    - If it's the first column (0), we set it to black.

16.     return cleaned
    - The function finishes and sends the repaired image back to the main program.

--------------------------------------------------------------------------------
BLOCK 3: VISUALIZATION (The Proof)
--------------------------------------------------------------------------------

17. def show_comparison(original, cleaned, label):
    - We define a function to show the popup window.
    - It takes the 'original' bad image, the 'cleaned' fixed image, and the 'label'.

18.     plt.figure(figsize=(8, 4))
    - We create a blank window that is 8 inches wide and 4 inches tall.

19.     plt.subplot(1, 2, 1); plt.imshow(original, cmap='gray'); plt.title(f"Bad: {label}")
    - We set up the Left side (1 row, 2 cols, position 1).
    - We draw the 'original' image in grayscale.
    - We add a title showing the label number.

20.     plt.subplot(1, 2, 2); plt.imshow(cleaned, cmap='gray'); plt.title("Fixed")
    - We set up the Right side (position 2).
    - We draw the 'cleaned' image.

21.     plt.tight_layout(); plt.show()
    - 'tight_layout': Adjusts spacing so titles don't overlap.
    - 'show': Pauses the code and pops the window up on your screen.

--------------------------------------------------------------------------------
BLOCK 4: MAIN EXECUTION (The Boss)
--------------------------------------------------------------------------------

22. def main():
    - We define the main starting point of the script.

23.     if len(sys.argv) < 7: sys.exit("Usage: python clean_data.py <dir> h w c <npz> <fix>")
    - Safety Check. We count the arguments typed in the terminal. 
    - If there are fewer than 7 (script name + 6 inputs), we stop and print help.

24.     img_dir, h, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    - We read the first 3 inputs: Folder path, Height, and Width.
    - We wrap 'int()' around h and w to convert the text "28" into the number 28.

25.     out_file, do_fix = sys.argv[5], int(sys.argv[6])
    - We read the output filename (data.npz) and the correction flag (0 or 1).

26.     print(f"Scanning {img_dir}...")
    - We print a message to the user confirming which folder we are working on.

27.     files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    - Complex Line:
      1. 'os.listdir': Gets all files in the folder.
      2. 'if f.endswith(".png")': Filters out anything that isn't a PNG image.
      3. 'sorted': Sorts them alphabetically (0-1.png, 0-2.png...) so order is safe.

28.     images, labels, shown = [], [], 0
    - We create empty lists to store the image data and labels later.
    - We set 'shown' to 0 (counter for how many popups we have displayed).

29.     for fname in files:
    - We loop through every single file found in line 27.

30.         path = os.path.join(img_dir, fname)
    - We build the full path (e.g., C:\Users\name\data\0-1.png).
    - This ensures the computer can find the file no matter what folder we are in.

31.         try:
    - We start a "Try Block". If any error happens inside, the code won't crash.

32.             label = int(fname.split("-")[0])
    - We split the filename "0-18.png" at the dash. 
    - We take the first part "0" and turn it into an integer (the Class Label).

33.             img = np.asarray(Image.open(path).convert("L"))
    - 1. 'Image.open': Opens the file.
    - 2. '.convert("L")': Forces it to Grayscale (Luminance).
    - 3. 'np.asarray': Turns the picture into a Matrix of numbers.

34.             if img.shape != (h, w): continue
    - If the image size is not 28x28, we use 'continue' to skip it and try the next file.

35.             if do_fix:
    - We check if the user typed '1' for the correction flag.

36.                 original = img.copy()
    - We save a backup of the current image to compare later.

37.                 img = clean_image(img)
    - We run the 'clean_image' function (from line 5) to remove white lines.

38.                 if shown < 3 and not np.array_equal(original, img):
    - We decide if we should show a popup.
    - "shown < 3": Have we shown fewer than 3 popups?
    - "not equal": Did the image actually change? (Was it corrupted?)

39.                     print(f"Visualizing fix for {fname}...")
    - We print a message to the terminal.

40.                     show_comparison(original, img, label)
    - We call the visualization function (from line 17).

41.                     shown += 1
    - We increase the counter so we don't show infinite popups.

42.             images.append(img)
    - We add the final (cleaned) image matrix to our big list.

43.             labels.append(label)
    - We add the correct label to our label list.

44.         except: continue
    - If any error occurred in the 'try' block (lines 31-43), we skip the file silently.

45.     np.savez(out_file, images=np.array(images), labels=np.array(labels))
    - We convert our lists into Numpy Arrays and save them into the .npz file.
    - This file contains the entire cleaned dataset.

46.     print(f"Done! Saved {len(images)} images to {out_file}")
    - We print a success message showing how many images were processed.

47. if __name__ == "__main__":
    - Standard Python check. Ensures 'main()' only runs if we execute this script directly.

48.     main()
    - Starts the program.
