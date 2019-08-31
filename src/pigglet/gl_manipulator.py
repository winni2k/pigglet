import numpy as np


class GLManipulator:
    def __init__(self, gls):
        self.gls = gls

    def normalize(self):
        """Revalue GLs so that every hom and het pair sum to 1 in real space"""
        rs_gls = np.exp(self.gls)
        rs_sum = rs_gls[:, :, 0] + rs_gls[:, :, 1]
        rs_gls[:, :, 0] /= rs_sum
        rs_gls[:, :, 1] /= rs_sum
        self.gls = np.log(rs_gls)
