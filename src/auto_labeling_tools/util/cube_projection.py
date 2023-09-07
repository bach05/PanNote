import numpy as np
from PIL import Image
from numpy import clip
from math import pi, atan2, hypot, floor


class CubeProjection:
    def __init__(self, imgIn, output_path):
        self.output_path = output_path
        self.imagin = imgIn
        self.sides = {
            'back': None,
            'left': None,
            'front': None,
            'right': None,
            'top': None,
            'bottom': None
        }

    def cube_projection(self):
        imgIn = self.imagin
        inSize = imgIn.size
        faceSize = int(inSize[0] / 4)

        FACE_NAMES = {
            0: 'back',
            1: 'left',
            2: 'front',
            3: 'right',
            4: 'top',
            5: 'bottom'
        }

        for face in range(6):
            # imgOut = Image.new('RGB', (faceSize, faceSize), 'black')
            imgOut = self.convertFace(imgIn,  (faceSize, faceSize), face)
            if self.output_path != '':
                imgOut.save(self.output_path + FACE_NAMES[face] + '.jpg')
            else:
                self.sides[FACE_NAMES[face]] = imgOut

    def convertFace(self, imgin, outsize, faceIdx):
        image = np.array(imgin)
        inSize = imgin.size
        facesize = outsize[0]

        coordinates = np.indices(outsize)  # zero is row coordinate one is col coordinates
        out = np.zeros((outsize[0], outsize[1], 3))

        # a = x coord; b = y coord
        if faceIdx == 0:
            out[:, :, 0] = -1.0
            out[:, :, 1] = 1.0 - (2 * coordinates[0] / facesize)
            out[:, :, 2] = 1.0 - (2 * coordinates[1] / facesize)
        elif faceIdx == 1:  # left
            out[:, :, 0] = (2 * coordinates[0] / facesize) - 1.0
            out[:, :, 1] = -1.0
            out[:, :, 2] = 1.0 - (2 * coordinates[1] / facesize)
        elif faceIdx == 2:  # front
            out[:, :, 0] = 1.0
            out[:, :, 1] = (2 * coordinates[0] / facesize) - 1.0
            out[:, :, 2] = 1.0 - (2 * coordinates[1] / facesize)
        elif faceIdx == 3:  # right
            out[:, :, 0] = 1 - (2 * coordinates[0] / facesize)
            out[:, :, 1] = 1.0
            out[:, :, 2] = 1.0 - (2 * coordinates[1] / facesize)
        elif faceIdx == 4:  # top
            out[:, :, 0] = (2 * coordinates[1] / facesize) - 1.0
            out[:, :, 1] = (2 * coordinates[0] / facesize) - 1.0
            out[:, :, 2] = 1.0
        elif faceIdx == 5:  # bottom
            out[:, :, 0] = 1.0 - (2 * coordinates[1] / facesize)
            out[:, :, 1] = (2 * coordinates[0] / facesize) - 1.0
            out[:, :, 2] = -1.0

        theta = np.arctan2(out[:, :, 1], out[:, :, 0])
        r = np.hypot(out[:, :, 1], out[:, :, 0])
        phi = np.arctan2(out[:, :, 2], r)

        uf = 0.5 * inSize[0] * (theta + pi) / pi
        vf = 0.5 * inSize[0] * (pi / 2 - phi) / pi

        ui = np.floor(uf)  # coord of pixel to bottom left
        vi = np.floor(vf)
        u2 = ui + 1  # coords of pixel to top right
        v2 = vi + 1
        mu = uf - ui  # fraction of way across pixel
        nu = vf - vi

        ui = (ui % inSize[0]).astype(int)
        vi = clip(vi, 0, inSize[1] - 1).astype(int)
        u2 = (u2 % inSize[0]).astype(int)
        v2 = clip(v2, 0, inSize[1] - 1).astype(int)

        A = image[vi, ui]
        B = image[vi, u2]
        C = image[v2, ui]
        D = image[v2, u2]

        r = A[:, :, 0] * (1 - mu) * (1 - nu) + B[:, :, 0] * (mu) * (1 - nu) + C[:, :, 0] * (1 - mu) * nu + D[:, :,
                                                                                                           0] * mu * nu
        g = A[:, :, 1] * (1 - mu) * (1 - nu) + B[:, :, 1] * (mu) * (1 - nu) + C[:, :, 1] * (1 - mu) * nu + D[:, :,
                                                                                                           1] * mu * nu
        b = A[:, :, 2] * (1 - mu) * (1 - nu) + B[:, :, 2] * (mu) * (1 - nu) + C[:, :, 2] * (1 - mu) * nu + D[:, :,
                                                                                                           2] * mu * nu

        r = np.round(r).astype(int)
        g = np.round(g).astype(int)
        b = np.round(b).astype(int)

        out[:, :, 0] = r
        out[:, :, 1] = g
        out[:, :, 2] = b

        return Image.fromarray(np.transpose(out, axes=(1, 0, 2)).astype(np.uint8))

def from_cube2panoramic(face, bnd):
    inSize = (1920*2, 960*2)
    a = 2.0 * float(bnd[0]) / 980
    b = 2.0 * float(bnd[1]) / 980
    if face == 'back':  # back
        (x, y, z) = (-1.0, 1.0 - a, 1.0 - b)
    elif face == 'left':  # left
        (x, y, z) = (a - 1.0, -1.0, 1.0 - b)
    elif face == 'front':  # front
        (x, y, z) = (1.0, a - 1.0, 1.0 - b)
    elif face == 'right':  # right
        (x, y, z) = (1.0 - a, 1.0, 1.0 - b)

    theta = atan2(y, x)  # range -pi to pi
    r = hypot(x, y)
    phi = atan2(z, r)  # range -pi/2 to pi/2

    # source img coords
    uf = 0.5 * 1920*2 * (theta + pi) / pi
    vf = 0.5 * 1920*2 * (pi / 2 - phi) / pi
    # Use bilinear interpolation between the four surrounding pixels
    ui = floor(uf)  # coord of pixel to bottom left
    vi = floor(vf)
    A = int(ui % inSize[0]), int(clip(vi, 0, inSize[1] - 1))
    return A