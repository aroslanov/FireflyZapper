import cv2
import numpy as np
import argparse
import OpenEXR
import Imath

def read_exr(filename):
    """Reads an EXR file and returns a NumPy array."""
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the channels into a NumPy array
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    red = np.frombuffer(exr_file.channel('R', FLOAT), dtype=np.float32).reshape(height, width)
    green = np.frombuffer(exr_file.channel('G', FLOAT), dtype=np.float32).reshape(height, width)
    blue = np.frombuffer(exr_file.channel('B', FLOAT), dtype=np.float32).reshape(height, width)

    return np.dstack((red, green, blue))

def write_exr(filename, data):
    """Writes a NumPy array to an EXR file."""
    # Ensure data is float32 and properly shaped
    data = data.astype(np.float32)
    if len(data.shape) == 2:
        data = np.dstack((data, data, data))
    elif data.shape[2] == 4:  # Handle RGBA images
        data = data[:, :, :3]  # Take only RGB channels

    # Create header dictionary
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    
    # Set the data window
    header['dataWindow'] = Imath.Box2i(Imath.V2i(0, 0), 
                                     Imath.V2i(data.shape[1] - 1, data.shape[0] - 1))
    header['displayWindow'] = header['dataWindow']

    # Set channel formats
    FLOAT = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, FLOAT) for c in "RGB"])

    # Ensure the data is contiguous in memory
    data = np.ascontiguousarray(data)
    
    out = OpenEXR.OutputFile(filename, header)

    # Write the channels separately
    R = data[:, :, 0].tobytes()
    G = data[:, :, 1].tobytes()
    B = data[:, :, 2].tobytes()

    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()

def main():
    parser = argparse.ArgumentParser(description='Firefly removal from images.')
    parser.add_argument('input', type=str, help='Input image file')
    parser.add_argument('output', type=str, help='Output image file')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for local statistics')
    parser.add_argument('--threshold', type=float, default=3.0, help='Z-score threshold for firefly detection')

    args = parser.parse_args()

    if args.input.endswith('.exr'):
        image = read_exr(args.input)
    else:
        image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image: {args.input}")

    if len(image.shape) == 3:
        # Color image: process each channel separately
        channels = cv2.split(image)
        processed_channels = []
        for chan in channels:
            processed_chan = process_channel(chan, args.window_size, args.threshold)
            processed_channels.append(processed_chan)
        result = cv2.merge(processed_channels)
    else:
        # Grayscale image
        result = process_channel(image, args.window_size, args.threshold)

    try:
        if args.output.endswith('.exr'):
            write_exr(args.output, result)
        else:
            cv2.imwrite(args.output, result)
    except Exception as e:
        raise RuntimeError(f"Failed to write output image: {str(e)}")


def process_channel(channel, window_size, threshold):
    # Ensure float32 for accurate computations
    channel_float = channel.astype(np.float32)

    # Compute local mean and std
    ksize = (window_size, window_size)
    mean = cv2.blur(channel_float, ksize)
    squared = cv2.blur(channel_float ** 2, ksize)
    variance = squared - (mean ** 2)

    # Ensure variance is non-negative to avoid invalid sqrt
    variance[variance < 0] = 0

    std = np.sqrt(variance)

    # Avoid division by zero
    std[std == 0] = 1e-6

    # Calculate z-scores
    z_scores = np.abs((channel_float - mean) / std)

    # Identify fireflies based on threshold
    is_firefly = z_scores > threshold

    # Apply median filter to the entire image
    median_filtered = cv2.medianBlur(channel.astype(np.float32), window_size)

    # Replace firefly pixels with median-filtered values
    result = np.where(is_firefly, median_filtered, channel_float)
    return result


if __name__ == "__main__":
    main()
