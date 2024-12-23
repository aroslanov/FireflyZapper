import cv2
import numpy as np
import argparse
import OpenEXR
import Imath
import os

# Constants for supported extensions and default values
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.exr']
WINDOW_SIZE_DEFAULT = 5
THRESHOLD_DEFAULT = 3.0

def read_exr(filename):
    """
    Reads an EXR file and returns a NumPy array containing the image data.

    Args:
        filename (str): Path to the EXR file.

    Returns:
        np.ndarray: A NumPy array representing the image in RGB format.
    """
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    red = np.frombuffer(exr_file.channel('R', FLOAT), dtype=np.float32).reshape(height, width)
    green = np.frombuffer(exr_file.channel('G', FLOAT), dtype=np.float32).reshape(height, width)
    blue = np.frombuffer(exr_file.channel('B', FLOAT), dtype=np.float32).reshape(height, width)

    return np.dstack((red, green, blue))

def write_exr(filename, data):
    """
    Writes a NumPy array to an EXR file.

    Args:
        filename (str): Path to the output EXR file.
        data (np.ndarray): A NumPy array representing the image in RGB format.
    """
    data = data.astype(np.float32)
    if len(data.shape) == 2:
        # If the input is a single channel, duplicate it to make it 3-channel
        data = np.dstack((data, data, data))
    elif data.shape[2] == 4:  
        # If the input has an alpha channel, remove it
        data = data[:, :, :3]

    header = OpenEXR.Header(data.shape[1], data.shape[0])
    header['dataWindow'] = Imath.Box2i(Imath.V2i(0, 0), 
                                     Imath.V2i(data.shape[1] - 1, data.shape[0] - 1))
    header['displayWindow'] = header['dataWindow']
    
    FLOAT = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, FLOAT) for c in "RGB"])
    
    data = np.ascontiguousarray(data)
    out = OpenEXR.OutputFile(filename, header)

    R = data[:, :, 0].tobytes()
    G = data[:, :, 1].tobytes()
    B = data[:, :, 2].tobytes()

    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()

def process_channel(channel, window_size, threshold):
    """
    Process a single channel to remove fireflies using local statistics and median filtering.

    Args:
        channel (np.ndarray): A NumPy array representing the image channel.
        window_size (int): The size of the window for computing local statistics.
        threshold (float): Z-score threshold for detecting fireflies.

    Returns:
        np.ndarray: Processed channel with fireflies removed.
    """
    channel_float = channel.astype(np.float32)

    ksize = (window_size, window_size)
    mean = cv2.blur(channel_float, ksize)
    squared = cv2.blur(channel_float ** 2, ksize)
    variance = squared - (mean ** 2)
    variance[variance < 0] = 0
    std = np.sqrt(variance)
    std[std == 0] = 1e-6

    z_scores = np.abs((channel_float - mean) / std)
    is_firefly = z_scores > threshold

    median_filtered = cv2.medianBlur(channel.astype(np.float32), window_size)
    result = np.where(is_firefly, median_filtered, channel_float)
    return result

def process_image(input_path, output_path, window_size, threshold):
    """
    Process an image to remove fireflies by processing each channel individually.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to the output image file.
        window_size (int): The size of the window for computing local statistics.
        threshold (float): Z-score threshold for detecting fireflies.
    """
    if input_path.endswith('.exr'):
        image = read_exr(input_path)
        original_dtype = np.float32
    else:
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        if image.dtype == np.uint16:
            original_dtype = np.uint16
        elif image.dtype == np.float32:
            original_dtype = np.float32
        elif image.dtype == np.uint8:
            original_dtype = np.uint8
        else:
            raise ValueError(f"Unsupported image type: {image.dtype}")

    if original_dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif original_dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    if len(image.shape) == 3:
        channels = cv2.split(image)
        processed_channels = [process_channel(chan, window_size, threshold) for chan in channels]
        result = cv2.merge(processed_channels)
    else:
        result = process_channel(image, window_size, threshold)

    if original_dtype == np.uint16:
        result = (result * 65535).astype(np.uint16)
    elif original_dtype == np.uint8:
        result = (result * 255).astype(np.uint8)

    try:
        if output_path.endswith('.exr'):
            write_exr(output_path, result)
        else:
            cv2.imwrite(output_path, result)
    except Exception as e:
        raise RuntimeError(f"Failed to write output image: {str(e)}")

def handle_same_directory(input_dir, output_dir):
    """
    Handle the case where input and output directories are the same.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.

    Returns:
        tuple: Updated paths for input and output directories and a prefix if applicable.
    """
    while input_dir == output_dir:
        print("Input and output directories are the same. This will overwrite source files.")
        user_choice = input("Enter 'n' for a new output directory, 'p' for an output file prefix, or 'o' to overwrite: ").strip().lower()
        
        if user_choice == 'n':
            output_dir = input("Enter new output directory path: ").strip()
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created new output directory: {output_dir}")
            except Exception as e:
                print(f"Failed to create directory: {str(e)}")
        elif user_choice == 'p':
            prefix = input("Enter the output file prefix: ").strip()
            if not prefix:
                print("Prefix cannot be empty. Please enter a valid prefix.")
            else:
                return input_dir, None, prefix
        elif user_choice == 'o':
            print("Overwriting files in the same directory.")
            return input_dir, input_dir, None
        else:
            print("Invalid choice. Please enter either 'n' (new), 'p' (prefix), or 'o' (overwrite).")
    
    return input_dir, output_dir, None

def get_output_path(input_path, output_dir, prefix=None):
    """
    Determine the output path based on the input file extension and any specified prefix.

    Args:
        input_path (str): Path to the input image file.
        output_dir (str): Path to the output directory.
        prefix (str, optional): Prefix for the output file name. Defaults to None.

    Returns:
        str: Full path to the output image file.
    """
    filename = os.path.basename(input_path)
    if prefix:
        filename = f"{prefix}{filename}"
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_processed{ext}"
    return os.path.join(output_dir, output_filename)

def main():
    parser = argparse.ArgumentParser(description='Firefly removal from images.')
    parser.add_argument('input', type=str, help='Input directory or image file')
    parser.add_argument('output', type=str, help='Output directory or image file')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE_DEFAULT, help='Window size for local statistics')
    parser.add_argument('--threshold', type=float, default=THRESHOLD_DEFAULT, help='Z-score threshold for firefly detection')

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    if os.path.isdir(input_path):
        input_dir = input_path
        output_dir = output_path

        input_dir, output_dir, prefix = handle_same_directory(input_dir, output_dir)

        for filename in os.listdir(input_dir):
            input_file_path = os.path.join(input_dir, filename)
            if any(input_file_path.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                output_filename = get_output_path(input_file_path, output_dir, prefix)
                print(f"Processing {input_file_path}...")
                process_image(input_file_path, output_filename, args.window_size, args.threshold)
    elif os.path.isfile(input_path):
        if any(input_path.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            input_dir = os.path.dirname(input_path)
            output_dir = os.path.dirname(output_path)

            if os.path.isdir(output_path):
                output_file_path = get_output_path(input_path, output_path)
            else:
                output_file_path = output_path

            print(f"Processing {input_path}...")
            process_image(input_path, output_file_path, args.window_size, args.threshold)
        else:
            raise ValueError("Invalid input file. Please provide a supported image file.")
    else:
        raise ValueError("Invalid input/output paths. Please provide valid directories or files.")

if __name__ == "__main__":
    main()