# Firefly Removal Script

## Overview

This repository contains a Python script (`zap.py`) designed to remove fireflies (hot pixels or bright spots) from images. The script utilizes local statistics and z-score thresholding to identify and replace firefly pixels with median-filtered values, effectively cleaning up the image.

## Features

- **Local Statistics**: Uses local mean and standard deviation to compute z-scores for each pixel.
- **Z-Score Thresholding**: Identifies fireflies based on a customizable z-score threshold.
- **Median Filtering**: Replaces detected firefly pixels with median-filtered values from their neighborhood.
- **Supports Both Grayscale and Color Images**: Processes each color channel separately to handle color images effectively.

## Installation

To use the `zap.py` script, you need Python installed on your system along with the required libraries. You can install these dependencies using pip:

```bash
pip install numpy opencv-python
```

Alternatively, you can create a virtual environment and install the dependencies there to avoid conflicts:

```bash
# Create a virtual environment
python -m venv firefly-removal-env

# Activate the virtual environment (Linux/Mac)
source firefly-removal-env/bin/activate

# Activate the virtual environment (Windows)
firefly-removal-env\Scripts\activate

# Install dependencies
pip install numpy opencv-python
```

## Usage

The script can be run from the command line with the following syntax:

```bash
python zap.py <input_image> <output_image> [--window_size <size>] [--threshold <value>]
```

### Arguments

- `<input_image>`: The path to the input image file (supported formats include JPEG, PNG, etc.).
- `<output_image>`: The path where the processed image will be saved.
- `--window_size`: (Optional) The size of the window used for local statistics calculations (default is 5x5).
- `--threshold`: (Optional) The z-score threshold for firefly detection (default is 3.0).

### Example

To remove fireflies from an image named `input.jpg` and save the result as `output.png`, you can run:

```bash
python zap.py input.jpg output.png --window_size 7 --threshold 2.5
```

## Running on Image Sequences

If you have a sequence of images and want to apply the firefly removal script to all of them, you can use a loop in the command line.

### For Windows

You can use PowerShell or Command Prompt. Here's an example using PowerShell:

```powershell
# Navigate to the directory containing your images
cd path\to\your\images

# Loop through each image file and apply zap.py
foreach ($file in Get-ChildItem -Filter *.jpg) {
    python ..\path\to\zap.py $file.FullName "$($file.BaseName)_processed.jpg" --window_size 7 --threshold 2.5
}
```

### For macOS/Linux

You can use a shell script or directly run the commands in the terminal:

```bash
# Navigate to the directory containing your images
cd path/to/your/images

# Loop through each image file and apply zap.py
for file in *.jpg; do
    python ../path/to/zap.py "$file" "${file%.jpg}_processed.jpg" --window_size 7 --threshold 2.5
done
```

## How It Works

1. **Reading the Image**: The script reads the input image using OpenCV.
2. **Processing Channels**: If the image is color, it splits the image into its RGB channels and processes each channel separately.
3. **Local Statistics Calculation**: For each channel (or the entire image if grayscale), local mean and standard deviation are calculated using a specified window size.
4. **Z-Score Computation**: The z-score for each pixel is computed based on the local statistics.
5. **Firefly Detection**: Pixels with z-scores above the threshold are identified as fireflies.
6. **Median Filtering**: A median filter is applied to the entire channel (or image).
7. **Replacement**: Detected firefly pixels are replaced with the corresponding values from the median-filtered result.
8. **Saving the Result**: The processed image is saved to the specified output path.

## Contributing

Contributions to this project are welcome! If you have any improvements, bug fixes, or additional features you'd like to add, feel free to fork the repository and submit a pull request. Please ensure your changes adhere to the project's coding standards and include appropriate documentation.

## License

This script is released under the MIT License. See the [LICENSE](LICENSE) file for more details.