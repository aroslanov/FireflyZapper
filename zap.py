import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Firefly removal from images.')
    parser.add_argument('input', type=str, help='Input image file')
    parser.add_argument('output', type=str, help='Output image file')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for local statistics')
    parser.add_argument('--threshold', type=float, default=3.0, help='Z-score threshold for firefly detection')

    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)

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

    cv2.imwrite(args.output, result)


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
    median_filtered = cv2.medianBlur(channel, window_size)

    # Replace firefly pixels with median-filtered values
    result = np.where(is_firefly, median_filtered, channel)
    return result


if __name__ == "__main__":
    main()
	
	
# This code should work for both grayscale and color images, handling different data types appropriately.

# I can further optimize it by ensuring that the window size is odd (since `cv2.medianBlur` requires an odd window
# size) and handling cases where the window size is even by adjusting it to the next odd number.

# Also, for very large images, this approach might be computationally intensive. In such cases, I can consider using
# multi-scale approaches or parallel processing to speed up the computations.

# Moreover, instead of using a fixed threshold, I could implement an adaptive threshold based on local statistics or
# other image features.

# Another potential improvement is to use more sophisticated methods for firefly detection and replacement, such as
# machine learning-based approaches or advanced image filtering techniques.

# However, for most practical purposes, this method should provide a reasonable starting point for removing
# fireflies from images.
