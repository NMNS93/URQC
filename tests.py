"""Unit tests for UltrasoundReverbQC"""

from UltrasoundReverbQC import USimg
import unittest
import numpy as np


class ImageLoadTest(unittest.TestCase):
    """Tests for image loading."""

    def setUp(self):
        self.jpg = USimg("images/20180220105300406.jpg")
        self.dcm = USimg("images/IMG_20131212_1_20.dcm")

    def test_readimg(self):
        # Raise IO error if invalid file extension
        with self.assertRaises(IOError):
            USimg('myfile')
        # Test that JPG and DICOM images are read to a numpy array
        self.assertIsInstance(self.jpg.img, np.ndarray)
        self.assertIsInstance(self.dcm.img, np.ndarray)

    def test_thresh(self):
        # Test that thresholding produces a binary-value array with only 0 and 255.
        thresh_j = self.jpg.threshold()
        thresh_d = self.dcm.threshold()
        unique_array = np.asarray([0, 255])
        self.assertTrue(np.all(unique_array == np.unique(thresh_j)))
        self.assertTrue(np.all(unique_array == np.unique(thresh_d)))

    def test_mask(self):
        # Test that the mask is not an empty array
        mask_j = self.jpg.maskimg()
        mask_d = self.jpg.maskimg()
        self.assertTrue(np.any(mask_j > 0))
        self.assertTrue(np.any(mask_d > 0))
        # Test that the mask contains fewer values than the input file
        self.assertTrue(np.sum(mask_j) < np.sum(self.jpg.img))
        self.assertTrue(np.sum(mask_d) < np.sum(self.dcm.img))
        pass


# TODO: Future tests


"""
Test the contour class points have coordinates where they should (leftmost is leftmost etc.)
Test that the top-right coordinates are not in the rotate data (stopped rotating) + test that 
    midpoint is in the rotate data (passed around image)
Test that the output files are written with the expected extensions.
"""


if __name__ == "__main__":
    unittest.main()
