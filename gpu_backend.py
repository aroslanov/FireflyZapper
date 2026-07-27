"""GPU acceleration backend for FireflyZapper.

CUDA and OpenCL implement the same local-statistics/conditional-median
algorithm as the NumPy/OpenCV fallback. GPU kernels currently support odd
windows up to 7x7; larger windows safely use the CPU implementation.
"""

import os
import tempfile
import warnings

import cv2
import numpy as np

# PyOpenCL/pytools otherwise choose user-profile cache paths that may be
# unwritable in packaged or managed environments.
_gpu_cache_root = os.path.join(tempfile.gettempdir(), "FireflyZapper-gpu-cache")
os.makedirs(_gpu_cache_root, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _gpu_cache_root)

try:
    import pycuda.driver as cuda_drv
    _CUDA_AVAILABLE = True
except ImportError:
    cuda_drv = None
    _CUDA_AVAILABLE = False

try:
    import pyopencl as cl
    _OPENCL_AVAILABLE = True
except ImportError:
    cl = None
    _OPENCL_AVAILABLE = False


GPU_MAX_WINDOW_SIZE = 7

_device_label = None
_device_stats = None
_device_error = None
_cuda_module = None
_opencl_context = None
_opencl_queue = None
_opencl_program = None


def _valid_gpu_window(window_size):
    return (
        isinstance(window_size, (int, np.integer))
        and 1 <= int(window_size) <= GPU_MAX_WINDOW_SIZE
        and int(window_size) % 2 == 1
    )


def _find_opencl_gpu():
    if not _OPENCL_AVAILABLE:
        return None
    for platform in cl.get_platforms():
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            return devices[0]
    return None


def get_device():
    """Return ``cuda``, ``opencl``, or ``cpu`` for the usable backend."""
    global _device_label, _device_error
    if _device_label is not None:
        return _device_label

    if _CUDA_AVAILABLE:
        try:
            cuda_drv.init()
            if cuda_drv.Device.count() > 0:
                _device_label = "cuda"
                return _device_label
        except Exception as exc:
            _device_error = f"CUDA detection failed: {exc}"

    if _OPENCL_AVAILABLE:
        try:
            if _find_opencl_gpu() is not None:
                _device_label = "opencl"
                return _device_label
        except Exception as exc:
            _device_error = f"OpenCL detection failed: {exc}"

    _device_label = "cpu"
    return _device_label


def _disable_gpu(label, exc):
    """Disable a backend after a runtime failure instead of silently retrying it."""
    global _device_label, _device_stats, _device_error
    _device_error = f"{label.upper()} runtime failed: {exc}"
    _device_label = "cpu"
    _device_stats = None
    warnings.warn(f"{_device_error}; using CPU fallback", RuntimeWarning, stacklevel=2)


def get_device_stats():
    global _device_stats
    if _device_stats is not None:
        return _device_stats

    label = get_device()
    if label == "cuda":
        try:
            device = cuda_drv.Device(0)
            attrs = device.get_attributes()
            _device_stats = {
                "name": device.name(),
                "memory": device.total_memory() // (1024 * 1024),
                "cores": attrs.get(cuda_drv.device_attribute.MULTIPROCESSOR_COUNT, "?"),
                "type": "cuda",
            }
            return _device_stats
        except Exception as exc:
            _disable_gpu("cuda", exc)
    elif label == "opencl":
        try:
            device = _find_opencl_gpu()
            _device_stats = {
                "name": device.name,
                "memory": device.global_mem_size // (1024 * 1024),
                "cores": device.max_compute_units,
                "type": "opencl",
            }
            return _device_stats
        except Exception as exc:
            _disable_gpu("opencl", exc)

    import multiprocessing
    import platform
    _device_stats = {
        "name": platform.processor() or "CPU",
        "memory": "system",
        "cores": multiprocessing.cpu_count(),
        "type": "cpu",
    }
    return _device_stats


_CUDA_KERNEL_SOURCE = r"""
extern "C" {
__device__ void insertion_sort(float *values, int count) {
    for (int i = 1; i < count; ++i) {
        float key = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > key) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = key;
    }
}

__global__ void remove_fireflies(
    const float *input, float *output, unsigned char *mask,
    int width, int height, int window_size, float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float values[49];
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;
    int radius = window_size / 2;
    for (int wy = -radius; wy <= radius; ++wy) {
        int sy = y + wy;
        if (sy < 0) sy = -sy;
        else if (sy >= height) sy = 2 * height - 2 - sy;
        for (int wx = -radius; wx <= radius; ++wx) {
            int sx = x + wx;
            if (sx < 0) sx = -sx;
            else if (sx >= width) sx = 2 * width - 2 - sx;
            float value = input[sy * width + sx];
            values[count++] = value;
            sum += value;
            sum_sq += value * value;
        }
    }

    float mean = sum / count;
    float variance = fmaxf(0.0f, sum_sq / count - mean * mean);
    float stddev = sqrtf(variance);
    if (stddev == 0.0f) stddev = 1e-6f;
    int index = y * width + x;
    bool detected = fabsf(input[index] - mean) / stddev > threshold;
    mask[index] = detected ? 1 : 0;
    if (detected) {
        insertion_sort(values, count);
        output[index] = values[count / 2];
    } else {
        output[index] = input[index];
    }
}
}
"""


_OPENCL_KERNEL_SOURCE = r"""
__kernel void remove_fireflies(
    __global const float *input, __global float *output, __global uchar *mask,
    int width, int height, int window_size, float threshold
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float values[49];
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;
    int radius = window_size / 2;
    for (int wy = -radius; wy <= radius; ++wy) {
        int sy = y + wy;
        if (sy < 0) sy = -sy;
        else if (sy >= height) sy = 2 * height - 2 - sy;
        for (int wx = -radius; wx <= radius; ++wx) {
            int sx = x + wx;
            if (sx < 0) sx = -sx;
            else if (sx >= width) sx = 2 * width - 2 - sx;
            float value = input[sy * width + sx];
            values[count++] = value;
            sum += value;
            sum_sq += value * value;
        }
    }

    float mean = sum / count;
    float variance = fmax(0.0f, sum_sq / count - mean * mean);
    float stddev = sqrt(variance);
    if (stddev == 0.0f) stddev = 1e-6f;
    int index = y * width + x;
    uchar detected = fabs(input[index] - mean) / stddev > threshold;
    mask[index] = detected;
    if (detected) {
        for (int i = 1; i < count; ++i) {
            float key = values[i];
            int j = i - 1;
            while (j >= 0 && values[j] > key) {
                values[j + 1] = values[j];
                --j;
            }
            values[j + 1] = key;
        }
        output[index] = values[count / 2];
    } else {
        output[index] = input[index];
    }
}
"""


def _get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        from pycuda.compiler import SourceModule
        cache_dir = os.path.join(tempfile.gettempdir(), "FireflyZapper-pycuda-cache")
        os.makedirs(cache_dir, exist_ok=True)
        _cuda_module = SourceModule(_CUDA_KERNEL_SOURCE, cache_dir=cache_dir)
    return _cuda_module


def _process_channel_cuda(channel_float, window_size, threshold, return_mask=False):
    import pycuda.autoinit  # noqa: F401 - creates the primary context
    import pycuda.gpuarray as gpuarray

    source = np.ascontiguousarray(channel_float, dtype=np.float32)
    height, width = source.shape
    d_input = gpuarray.to_gpu(source)
    d_output = gpuarray.empty_like(d_input)
    d_mask = gpuarray.empty(source.shape, dtype=np.uint8)
    kernel = _get_cuda_module().get_function("remove_fireflies")
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16)
    kernel(
        d_input, d_output, d_mask,
        np.int32(width), np.int32(height), np.int32(window_size), np.float32(threshold),
        block=block, grid=grid,
    )
    result = d_output.get()
    mask = d_mask.get().astype(bool) if return_mask else None
    return result, mask


def _get_opencl_runtime():
    global _opencl_context, _opencl_queue, _opencl_program
    if _opencl_program is None:
        device = _find_opencl_gpu()
        if device is None:
            raise RuntimeError("No OpenCL GPU is available")
        _opencl_context = cl.Context([device])
        _opencl_queue = cl.CommandQueue(_opencl_context)
        _opencl_program = cl.Program(_opencl_context, _OPENCL_KERNEL_SOURCE).build()
    return _opencl_context, _opencl_queue, _opencl_program


def _process_channel_opencl(channel_float, window_size, threshold, return_mask=False):
    source = np.ascontiguousarray(channel_float, dtype=np.float32)
    height, width = source.shape
    context, queue, program = _get_opencl_runtime()
    flags = cl.mem_flags
    d_input = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=source)
    d_output = cl.Buffer(context, flags.WRITE_ONLY, source.nbytes)
    d_mask = cl.Buffer(context, flags.WRITE_ONLY, source.size)
    global_size = (((width + 15) // 16) * 16, ((height + 15) // 16) * 16)
    program.remove_fireflies(
        queue, global_size, (16, 16), d_input, d_output, d_mask,
        np.int32(width), np.int32(height), np.int32(window_size), np.float32(threshold),
    )
    result = np.empty_like(source)
    cl.enqueue_copy(queue, result, d_output)
    mask = None
    if return_mask:
        mask_bytes = np.empty(source.shape, dtype=np.uint8)
        cl.enqueue_copy(queue, mask_bytes, d_mask)
        mask = mask_bytes.astype(bool)
    queue.finish()
    return result, mask


def process_channel_cpu(channel, window_size, threshold, return_mask=False):
    channel_float = np.asarray(channel, dtype=np.float32)
    ksize = (window_size, window_size)
    mean = cv2.blur(channel_float, ksize)
    mean_sq = cv2.blur(channel_float * channel_float, ksize)
    variance = np.maximum(mean_sq - mean * mean, 0)
    std = np.sqrt(variance)
    std[std == 0] = 1e-6
    mask = np.abs((channel_float - mean) / std) > threshold
    half = window_size // 2
    padded = np.pad(channel_float, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (window_size, window_size)
    )
    median = np.median(windows, axis=(-2, -1))
    result = np.where(mask, median, channel_float)
    return (result, mask) if return_mask else result


def process_channel_gpu(
    channel, window_size, threshold, device_label=None, return_mask=False
):
    """Process one channel, safely and truthfully falling back to the CPU."""
    channel_float = np.asarray(channel, dtype=np.float32)
    label = device_label or get_device()
    # A prior channel may have disabled this backend during the same RGB frame.
    if device_label is not None and _device_error and get_device() == "cpu":
        label = "cpu"
    if label == "cpu" or not _valid_gpu_window(window_size):
        return process_channel_cpu(
            channel_float, window_size, threshold, return_mask=return_mask
        )

    try:
        if label == "cuda":
            result, mask = _process_channel_cuda(
                channel_float, window_size, threshold, return_mask
            )
        elif label == "opencl":
            result, mask = _process_channel_opencl(
                channel_float, window_size, threshold, return_mask
            )
        else:
            raise ValueError(f"Unknown device backend: {label}")
        return (result, mask) if return_mask else result
    except Exception as exc:
        _disable_gpu(label, exc)
        return process_channel_cpu(
            channel_float, window_size, threshold, return_mask=return_mask
        )


def process_image_gpu(
    image, window_size, threshold, device_label=None, return_mask=False
):
    """Process an image and optionally return its combined detection mask."""
    image_float = np.asarray(image, dtype=np.float32)
    label = device_label or get_device()
    if image_float.ndim == 2:
        return process_channel_gpu(
            image_float, window_size, threshold, label, return_mask
        )

    results = []
    masks = []
    for channel in cv2.split(image_float):
        if return_mask:
            result, mask = process_channel_gpu(
                channel, window_size, threshold, label, True
            )
            masks.append(mask)
        else:
            result = process_channel_gpu(
                channel, window_size, threshold, label, False
            )
        results.append(result)
    merged = cv2.merge(results)
    if return_mask:
        return merged, np.any(masks, axis=0)
    return merged


def get_device_status():
    label = get_device()
    stats = get_device_stats()
    if label in {"cuda", "opencl"}:
        return (
            f"GPU: {label.upper()} — {stats['name']} "
            f"({stats['cores']} compute units, {stats['memory']} MB) | "
            "Acceleration: Active"
        )
    reason = f" — {_device_error}" if _device_error else ""
    return (
        f"CPU fallback — {stats['name']} ({stats['cores']} cores) | "
        f"Acceleration: Disabled{reason}"
    )


def is_gpu_active():
    return get_device() in {"cuda", "opencl"}


def reload_device_status():
    global _device_label, _device_stats, _device_error
    _device_label = None
    _device_stats = None
    _device_error = None
    get_device()
    return get_device_stats()


if __name__ == "__main__":
    print(get_device_status())
