# Firefly Removal Script

## Overview

This repository contains a Python script (`zap.py`) designed to remove fireflies (hot pixels or bright spots) from images. The script utilizes local statistics and z-score thresholding to identify and replace firefly pixels with median-filtered values, effectively cleaning up the image.

## Features

- **Local Statistics**: Uses local mean and standard deviation to compute z-scores for each pixel.
- **Z-Score Thresholding**: Identifies fireflies based on a customizable z-score threshold.
- **Median Filtering**: Replaces detected firefly pixels with median-filtered values from their neighborhood.
- **Supports Grayscale, Color Images, and EXR Format**: Processes each color channel separately to handle color images effectively. Additionally, it supports high dynamic range (HDR) images in the EXR format.

## Installation

To use the `zap.py` script, you need Python installed on your system along with the required libraries. You can install these dependencies using pip:

```bash
pip install numpy opencv-python openexr
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
pip install numpy opencv-python openexr
```

## Usage

The script can be run from the command line with the following syntax:

```bash
python zap.py <input> <output> [--window_size <size>] [--threshold <value>]
```

### Arguments

- `<input>`: The path to the input image file or directory. Supported formats include JPEG, PNG, BMP, and EXR for images.
  - If a single image is provided, it will be processed and saved as specified by the output argument.
  - If a directory is provided, all supported images in that directory will be processed, and the results will be saved to the output directory or prefixed with the specified prefix.

- `<output>`: The path where the processed image(s) will be saved. This can be either:
  - A file path if a single input image is provided.
  - A directory path if an input directory is provided. If this is the same as the input directory, you will be prompted to choose between creating a new output directory, adding a prefix to the output files, or overwriting the source files.

- `--window_size <size>`: (Optional) The size of the window used for local statistics calculations. Must be an odd integer (default is 5). Larger windows can smooth out more noise but may also remove smaller details.

- `--threshold <value>`: (Optional) The z-score threshold for firefly detection. Pixels with z-scores above this value are considered fireflies and will be replaced (default is 3.0).

### Example

To remove fireflies from an image named `input.jpg` and save the result as `output.png`, you can run:

```bash
python zap.py input.jpg output.png --window_size 7 --threshold 2.5
```

To process all images in a directory named `images` and save the results in a new directory named `processed_images`, you can run:

```bash
python zap.py images processed_images --window_size 9 --threshold 3.5
```

If you want to add a prefix `clean_` to each output file name, you can run:

```bash
python zap.py images images --prefix clean_
```

This will process all supported images in the `images` directory and save them back into the same directory with the prefix `clean_`.

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

1. **Reading the Image**: The script reads the input image using OpenCV or `OpenEXR` for EXR files.
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