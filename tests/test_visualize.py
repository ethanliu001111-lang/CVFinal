import unittest

import numpy as np

from demo.src.visualize import smpl_y_up_to_matplotlib_z_up


class VisualizeTests(unittest.TestCase):
    def test_maps_smpl_y_axis_to_matplotlib_z_axis(self):
        verts = np.array([
            [0.0, -1.0, 0.2],
            [0.0,  2.0, 0.2],
            [0.5,  0.0, 1.2],
        ])

        oriented = smpl_y_up_to_matplotlib_z_up(verts)

        self.assertEqual(oriented.shape, verts.shape)
        self.assertAlmostEqual(np.ptp(oriented[:, 2]), np.ptp(verts[:, 1]))
        self.assertAlmostEqual(np.ptp(oriented[:, 1]), np.ptp(verts[:, 2]))


if __name__ == "__main__":
    unittest.main()
