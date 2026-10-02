import unittest

import numpy as np

from frigate.util.model import post_process_yolo_end2end


class TestYoloEnd2EndPostProcess(unittest.TestCase):
    """YOLO26 (and other end2end exports) emit (1, max_det, 6) rows of
    [x1, y1, x2, y2, conf, class_id] in pixel coordinates."""

    def _tensor(self, rows, max_det=300):
        out = np.zeros((1, max_det, 6), dtype=np.float32)
        out[0, : len(rows)] = rows
        return out

    def test_decodes_boxes_into_frigate_layout(self):
        tensor = self._tensor([[32, 64, 160, 320, 0.9, 2]])
        detections = post_process_yolo_end2end([tensor], 320, 640)

        self.assertEqual(detections.shape, (20, 6))
        np.testing.assert_allclose(
            detections[0], [2, 0.9, 0.1, 0.1, 0.5, 0.5], rtol=1e-6
        )
        self.assertTrue(np.all(detections[1:] == 0))

    def test_padding_and_low_confidence_rows_are_dropped(self):
        tensor = self._tensor(
            [
                [0, 0, 10, 10, 0.95, 0],
                [0, 0, 10, 10, 0.39, 1],
                [0, 0, 10, 10, 0.4, 1],
            ]
        )
        detections = post_process_yolo_end2end([tensor], 100, 100)

        self.assertEqual(int(np.count_nonzero(detections[:, 1])), 1)
        self.assertEqual(detections[0, 0], 0)

    def test_results_are_capped_at_twenty_and_sorted(self):
        rows = [[0, 0, 10, 10, 0.5 + i * 0.01, i % 3] for i in range(30)]
        detections = post_process_yolo_end2end([self._tensor(rows)], 100, 100)

        self.assertEqual(int(np.count_nonzero(detections[:, 1])), 20)
        self.assertAlmostEqual(float(detections[0, 1]), 0.79, places=5)
        self.assertTrue(np.all(np.diff(detections[:, 1]) <= 0))

    def test_coordinates_are_clipped_to_the_frame(self):
        tensor = self._tensor([[-5, -5, 330, 330, 0.8, 0]])
        detections = post_process_yolo_end2end([tensor], 320, 320)

        np.testing.assert_allclose(detections[0, 2:], [0, 0, 1, 1])

    def test_accepts_a_bare_array_and_a_squeezed_array(self):
        rows = [[0, 0, 50, 50, 0.7, 4]]
        expected = post_process_yolo_end2end([self._tensor(rows)], 100, 100)

        np.testing.assert_array_equal(
            post_process_yolo_end2end(self._tensor(rows), 100, 100), expected
        )
        np.testing.assert_array_equal(
            post_process_yolo_end2end(self._tensor(rows)[0], 100, 100), expected
        )

    def test_a_raw_head_is_rejected_instead_of_misread(self):
        raw_head = np.random.rand(1, 84, 8400).astype(np.float32)
        detections = post_process_yolo_end2end([raw_head], 640, 640)

        self.assertTrue(np.all(detections == 0))
