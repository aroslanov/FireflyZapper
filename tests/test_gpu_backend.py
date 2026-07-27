import unittest
from unittest import mock

import numpy as np

import gpu_backend as backend


class GPUBackendTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).random((24, 32), dtype=np.float32)

    def test_cpu_result_and_mask_are_consistent(self):
        result, mask = backend.process_channel_cpu(
            self.image, 5, 3.0, return_mask=True
        )
        result_without_mask = backend.process_channel_cpu(
            self.image, 5, 3.0, return_mask=False
        )
        np.testing.assert_array_equal(result, result_without_mask)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, self.image.shape)

    def test_oversized_window_never_launches_gpu_kernel(self):
        with mock.patch.object(
            backend, "_process_channel_cuda",
            side_effect=AssertionError("unsafe kernel launch"),
        ):
            result = backend.process_channel_gpu(
                self.image, 9, 3.0, device_label="cuda"
            )
        expected = backend.process_channel_cpu(self.image, 9, 3.0)
        np.testing.assert_array_equal(result, expected)

    def test_runtime_failure_disables_backend_and_returns_cpu_result(self):
        old_label = backend._device_label
        old_stats = backend._device_stats
        old_error = backend._device_error
        try:
            backend._device_label = "cuda"
            with mock.patch.object(
                backend, "_process_channel_cuda", side_effect=RuntimeError("boom")
            ), self.assertWarns(RuntimeWarning):
                result = backend.process_channel_gpu(
                    self.image, 5, 3.0, device_label="cuda"
                )
            expected = backend.process_channel_cpu(self.image, 5, 3.0)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(backend.get_device(), "cpu")
            self.assertIn("boom", backend.get_device_status())
        finally:
            backend._device_label = old_label
            backend._device_stats = old_stats
            backend._device_error = old_error

    def test_image_mask_combines_channel_masks(self):
        rgb = np.dstack((self.image, self.image, self.image))
        result, mask = backend.process_image_gpu(
            rgb, 5, 3.0, device_label="cpu", return_mask=True
        )
        self.assertEqual(result.shape, rgb.shape)
        self.assertEqual(mask.shape, self.image.shape)
        self.assertEqual(mask.dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
