#!/usr/bin/env python3
"""
UltrasoundReverbQC.py
Calculate average grayscale pixel intensity values from ultrasound in-air reverberation patterns.

Usage: 
python3 UltrasoundReverbQC.py -i INPUT [-o OUTPREFIX]

Author(s) : Nana Mensah <Nana.mensah1@nhs.net>
Created : 11 April 2018
"""
from matplotlib import pyplot as plt
import argparse
import csv
import cv2
import itertools
import numpy as np
import os
import pydicom

class Contour(object):
    """OpenCV contour object for the Ultrasound reverberation pattern.

    Args:
        img(array): Binary image produced by thresholding.
    """

    def __init__(self, img):
        """Initialise class with a binary image produced by thresholding."""
        self.image = img
        self.UScontour = self.find_contour()

    @staticmethod
    def aspect_ratio(cont):
        """Calculates the aspect ration of an OpenCV contour object."""
        x, y, w, h = cv2.boundingRect(cont)
        return float(w) / h

    def find_contour(self):
        """Return the OpenCV contour object containing the ultrasound reverbation pattern.
        From all possible contours, this selection is based on the following criteria:
        - A contour area of >200,000 units.
        - The first contour with an aspect ratio between 2 and 5.
        """

        # Call OpenCV findContours to generate a list of contour objects
        im, contours, heirarchy = cv2.findContours(self.image, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)

        # Create a list of tuples for each contour (contour, aspect_ratio, contourArea),
        # arranged by countourArea in descending order
        cont_list = sorted([(cont, self.aspect_ratio(cont), cv2.contourArea(cont)) for cont in contours],
                           key=lambda x: x[2], reverse=True)

        # Filter sorted contour list for contours with aspect ratio < 10 (to ignore vendor headband
        # which lies ~17) and contourArea > 200000. This filter selects high-quality reverberation
        # patterns that span the majority of the image.
        large_area_bool = list(filter(lambda x: (x[1] < 10) and (x[2] > 200000), cont_list))

        # Filter sorted contour list for contours with aspect ratio to between 2 and 5. The aspect
        # ratio for reverbration patterns in test images were approximately 3.33.
        asp_ratio_bool = list(filter(lambda x: (x[1] > 2) and (x[1] < 5), cont_list))

        # Return the contour object
        if large_area_bool:
            return large_area_bool[0][0]
        else:
            return asp_ratio_bool[0][0]

    @staticmethod
    def intersect(point_dict):
        """
        Returns the point of intersection of the lines passing through a1,a2 and b1,b2.
        """
        a1, a2, b1, b2 = [point_dict[i] for i in ['top-left', 'left', 'top-right', 'right']]
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # if lines are parallel
            return float('inf'), float('inf')
        return int(x / z), int(y / z)

    @property
    def points(self):
        """Dict containing corner points of the contour (left, right, top-left, top-right), midpoint
        (mid) and the edge line intersect (ins).
        """

        # Initialise empty dictionary for points
        point_dict = dict()

        # Get contour array
        cnt = self.UScontour

        # Find the moments in the reverb image contour for calculating the midpoint
        M = cv2.moments(cnt)
        # Find the midpoint of the contour
        point_dict['mid'] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        # Set the left-most point of the contour (bottom-left corner of the reverb image)
        point_dict['left'] = tuple(cnt[cnt[:, :, 0].argmin()][0])
        # Find the right-most point of the contour (bottom-right corner of the reverb image)
        point_dict['right'] = tuple(cnt[cnt[:, :, 0].argmax()][0])

        # Set x-coordinate of contour midpoint
        midx = point_dict['mid'][0]
        # Filter contour points left and right of midpoint
        cnt_left = cnt[np.where(cnt[:, :, 0] < midx)]
        cnt_right = cnt[np.where(cnt[:, :, 0] > midx)]

        # Set top-left and top-right
        point_dict['top-left'] = tuple(cnt_left[cnt_left[:, 1].argmin()])
        point_dict['top-right'] = tuple(cnt_right[cnt_right[:, 1].argmin()])

        # Set intersect of edge lines
        point_dict['ins'] = self.intersect(point_dict)

        return point_dict


class USimg(object):
    """Class object for ultrasound (curved array) reverb image processing.

    Args:
        infile (str): Input file in DICOM or JPEG format.

    Attributes:
        thresh (numpy.ndarray): Binary image produced using thresholding.
        maskimg(np.ndarray) : Image with reverberation pattern isolated by segmentation.
        contour(cv2 contour object) : OpenCV contour object.
    """

    def __init__(self, infile):
        self.img = cv2.cvtColor(self.read(infile), cv2.COLOR_BGR2GRAY)
        self.thresh = self.threshold()
        cont = Contour(self.thresh)
        self.contour = cont.UScontour
        self.points = cont.points
        self.mask = self.maskimg()
        self.coords, self.data, self.avgs = self.reverb_data()

    @staticmethod
    def read(infile):
        """Return a pixel array from an input image file, including DICOM format"""
        # Raise error file input file does not exist
        if not os.path.isfile(infile):
            raise IOError('Invalid input file.')

        if infile.endswith('.dcm'):
            return pydicom.dcmread(infile).pixel_array
        else:
            return cv2.imread(infile)

    def threshold(self):
        """Return an array containing the binary input image after thresholding using Otsu's binarization."""
        # Apply Gaussian blurring to slightly increase values around pixels with high intensity
        blur = cv2.GaussianBlur(self.img, (5, 5), 0)
        # Apply histogram equalisation to spread and amplify pixel values
        equ = cv2.equalizeHist(blur)
        # Use OTSUs thresholding algorithm to find the optimum threshold value
        # Here, retVal is the optimum threshold value, or the user default if not found
        ret, th = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Return binary image
        return th

    def maskimg(self):
        """Returns an array containing the original image masked to display the reverberation
        pattern selected by USimg.contour()."""
        # Create an empty numpy array with the same dimensions as the input image
        image = self.img
        mask = np.zeros(image.shape, np.uint8)
        # Obtain the contour for the reverberation pattern
        cont = self.contour
        # Apply the contour object points to the blank mask
        cv2.drawContours(mask, [cont], 0, 255, -1)
        # Expand the mask image boundary using morphological dilation, to capture rough image edges.
        kernel = np.ones((8, 8), np.uint8)
        dilate_mask = cv2.dilate(mask, kernel, iterations=2)
        # Return the mask image.
        return cv2.bitwise_and(image, image, mask=dilate_mask)

    def reverb_data(self):
        """Return a tuple containing two data structures resulting from reading the reverb image:
        us_coords : A list of radians and coordinates. All coordinates are from the line parallel to
        the contour left edge, rotated by the radians value about the edge line intersect.
        us_data : A list of radians and pixel values. All pixel values originate from us_coords.
        """

        # Set the radians value to rotate by
        RADS = 0.004

        def rotate(point, angle=(-1 * RADS), origin=self.points['ins']):
            """Rotate a point counterclockwise by a given angle (radians) around a given origin."""
            ox, oy = origin
            px, py = point
            qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
            qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
            return qx, qy

        def coord_filter(coords):
            """Return the pixel value for a coordinate if it falls within the image array.
            Returns None if outside of array."""
            y_max, x_max = self.img.shape
            x, y = coords
            if all([(y <= y_max), (y >= 0), (x <= x_max), (x >= 0)]):
                return self.mask[int(y), int(x)]  # Note img[y,x] slicing versus (x,y) coordinate.

        def avg(data):
            """Return the average of a list of pixel values."""
            array = np.asarray(data[1])
            return data[0], np.mean(array[np.nonzero(array)])

        def sweep(start_coords):
            """Take a list of coordinates and apply the rotate() function sequentially.
            Yield the radians value (from start) and coordinates of the points.
            """
            nrads = 0
            mcords = start_coords
            while sum(filter(lambda x: x is not None, map(coord_filter, mcords))) > 0:
                nrads += RADS
                mcords = list(map(rotate, mcords))
                yield nrads, mcords

        # Get cooefficients of the left edge line equation
        zip_ledge = list(zip(self.points['left'], self.points['top-left']))
        gradient, y_int = (np.polyfit(zip_ledge[0], zip_ledge[1], 1))
        # Get list of left edge coordinates (x, int(y))
        ledge_coords = [(x, (gradient * x + y_int)) for x in range(self.img.shape[1])]

        # Rotate the list of left edge coordinates sequentially by the RADS value across the
        # reverb image. This moves the "line" of coordinates from the left edge to the right.
        # Store the radians and each coordinate in the us_coords list.
        us_coords = [[0, ledge_coords]] + [list(coords) for coords in sweep(ledge_coords)]

        # Create list of rads and pixel values for the coordinates in us_coords. All coordinates
        # returning None with coord_filter are dropped from the list.
        us_data = [(rads, list(filter(None.__ne__, map(coord_filter, coords))))
                   for rads, coords in us_coords]

        us_avgs = list(map(avg, us_data))

        return us_coords, us_data, us_avgs

    def depth_data(self):
        """Create data for depth calculations and plots. Emulates reading linear array horizontally.
        """
        def strip_list(inlist):
            """Remove leading and trailing 0 values from a list. This removes the empty pixels read
            from the masked image."""
            strip = inlist.copy()
            for index in (0, -1):
                while strip and (strip[index] == 0):
                    strip.pop(index)
            return strip

        # Get a list of lists of pixel values with leading and trailing 0s filtered.
        # This list is reversed to account for reading of the line in reverse in self.reverb_data()
        pix_list = [list(reversed(strip_list(i[1]))) for i in self.data]

        # Transpose the nested lists such that each value is stored with others of the same index
        transp = list(itertools.zip_longest(*pix_list, fillvalue=0))

        # Get the average for each value.
        depth_avg = [ np.mean(i) for i in transp ]

        # Get the maximum depth and rows read
        us_depth = list(set([len(i) for i in transp]))[0]
        rows = len(depth_avg)

        return {'data': transp, 'avg': depth_avg, 'depth': us_depth, 'rows': rows}

    def write(self, prefix):
        """Write out data and plots."""
        # Create output directory if it does not exist
        dirlist = ['data', 'plots']
        for i in dirlist:
            if not os.path.isdir(i):
                os.mkdir(i)

        # Write masked image
        cv2.imwrite("plots/" + prefix + "_urqc_img.png", self.mask)
        # Write reverb pixel values
        with open("data/" + prefix + "_urqc_data.csv", 'w+') as f:
            writer = csv.writer(f)
            writer.writerows(self.data)

        # Write data plot (averages)
        rads, avgs = zip(*self.avgs)
        plt.plot(rads, avgs)
        plt.savefig(("plots/" + prefix + "_urqc_plot.png"))
        plt.gcf().clear()

        # Write average raw data
        with open("data/" + prefix + "_urqc_avgs.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.avgs)

        # Write depth plot (averages)
        depth_data = self.depth_data()
        plt.plot(depth_data['avg'], 'r,-')
        plt.savefig(("plots/" + prefix + "_urqc_horiz.png"))
        plt.gcf().clear()

        # Write depth plot data
        with open("data/" + prefix + "_urqc_depthdata.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows([[depth_data['depth']], [depth_data['rows']]])
            writer.writerows([[depth_data['data']]])


if __name__ == "__main__":

    # Configure argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='{} v1.0'.format(parser.prog))
    parser.add_argument('input', type=str, help='*.jpg, *.png, *.dcm files', metavar='INPUT', nargs='+')
    parser.add_argument('-o', '--outprefix', type=str, help='Output file prefix')
    opts, args = parser.parse_known_args()

    for input in opts.input:
        # Create an instance of USimg with input
        urqc = USimg(input)
        # Set filename as outprefix for use as default
        path_no_ext = os.path.splitext(input)[0]
        outprefix = os.path.basename(path_no_ext)

        # Write out data and plots from the reverb image
        if opts.outprefix:
            urqc.write(opts.outprefix)
        else:
            urqc.write(outprefix)
