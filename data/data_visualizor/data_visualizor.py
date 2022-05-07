import numpy as np
import mayavi.mlab

def lidar_visualizor(points, method):
    """
        Use different visualize methods to denoise lidar data.
        Args:
            points: np.array
            method: str
        return:
            0
        Reference:
            https://github.com/strawlab/python-pcl
    """
    if method == 'mayavi':
        x = points[:, 0]  # x position of point
        y = points[:, 1]  # y position of point
        z = points[:, 2]  # z position of point

        r = points[:, 3]  # reflectance value of point
        d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

        degr = np.degrees(np.arctan(z / d))

        vals = 'height'
        if vals == "height":
            col = z
        else:
            col = d

        fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        mayavi.mlab.points3d(x, y, z,
                             col,  # Values used for Color
                             mode="point",
                             colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                             # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                             figure=fig,
                             )

        mayavi.mlab.show()

    return 0

