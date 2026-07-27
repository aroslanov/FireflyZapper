import unittest
from unittest import mock

import numpy as np

import zap_gui


class GUIProcessingTests(unittest.TestCase):
    def test_sequence_mode_can_skip_mask(self):
        image = np.zeros((8, 10, 3), dtype=np.float32)
        expected = np.ones_like(image)
        with mock.patch.object(
            zap_gui, "process_image_gpu", return_value=expected
        ) as process:
            result, mask = zap_gui.process_image_full(
                image, 5, 3.0, use_gpu=True, return_mask=False
            )
        np.testing.assert_array_equal(result, expected)
        self.assertIsNone(mask)
        self.assertFalse(process.call_args.kwargs["return_mask"])

    def test_preview_requests_backend_mask_once(self):
        image = np.zeros((8, 10, 3), dtype=np.float32)
        expected = np.ones_like(image)
        expected_mask = np.zeros(image.shape[:2], dtype=bool)
        with mock.patch.object(
            zap_gui, "process_image_gpu", return_value=(expected, expected_mask)
        ) as process:
            result, mask = zap_gui.process_image_full(
                image, 5, 3.0, use_gpu=True, return_mask=True
            )
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertEqual(process.call_count, 1)
        self.assertTrue(process.call_args.kwargs["return_mask"])


if __name__ == "__main__":
    unittest.main()
