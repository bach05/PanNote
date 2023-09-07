from PIL import Image
from math import pi, sin, cos, tan, atan2, hypot, floor
from numpy import clip
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import time

class CubeProjection:
    def __init__(self, imgIn, output_path):
        self.output_path = output_path
        self.imagin = imgIn
        self.sides = []

    def cube_projection(self, face_id=None, img_id="img"):
        imgIn = self.imagin
        inSize = imgIn.size
        faceSize = int(inSize[0]/4)
        FACE_NAMES = {
            0: 'back',
            1: 'left',
            2: 'front',
            3: 'right',
            4: 'top',
            5: 'bottom'
        }

        FACE_NAMES_RED = {
            0: 'back',
            1: 'left',
            2: 'front',
            3: 'right',
        }

        if face_id is None:
            img_sides = []
            start_time = time.time()
            for face in range(4):
                imgOut = self.convertFace(imgIn, (faceSize, faceSize), face)
                img_sides.append(imgOut)
                if self.output_path != '':
                    save_path = os.path.join(self.output_path, FACE_NAMES_RED[face]+img_id+'.jpg')
                    print("SAVED: ", save_path)
                    imgOut.save(save_path)
                else:
                    self.sides.append({FACE_NAMES[face]: None})
            end_time = time.time()
            elapsed_time = end_time - start_time
            #print("Elapsed time:", elapsed_time, "seconds")
            return img_sides
        else:
            face = face_id
            imgOut = self.convertFace(imgIn, (faceSize, faceSize), face)
            if self.output_path != '':
                save_path = os.path.join(self.output_path,FACE_NAMES[face],img_id+'.jpg')
                print("SAVED: ", save_path)
                imgOut.save(save_path)
            else:
                self.sides.append({FACE_NAMES[face]: None})
            return imgOut


    def convertFace(self, imgin, outsize, faceIdx):
        image = np.array(imgin)
        inSize = imgin.size
        facesize = outsize[0]

        coordinates = np.indices(outsize) # zero is row coordinate one is col coordinates
        out = np.zeros((outsize[0], outsize[1], 3))

        # a = x coord; b = y coord
        if faceIdx == 0:
            out[:,:,0] = -1.0
            out[:,:,1] = 1.0 - (2 * coordinates[0] / facesize)
            out[:,:,2] = 1.0 - (2 * coordinates[1] / facesize)
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

        r = A[:, :, 0] * (1 - mu) * (1 - nu) + B[:, :, 0] * (mu) * (1 - nu) + C[:, :, 0] * (1 - mu) * nu + D[:, :, 0] * mu * nu
        g = A[:, :, 1] * (1 - mu) * (1 - nu) + B[:, :, 1] * (mu) * (1 - nu) + C[:, :, 1] * (1 - mu) * nu + D[:, :, 1] * mu * nu
        b = A[:, :, 2] * (1 - mu) * (1 - nu) + B[:, :, 2] * (mu) * (1 - nu) + C[:, :, 2] * (1 - mu) * nu + D[:, :, 2] * mu * nu

        r = np.round(r).astype(int)
        g = np.round(g).astype(int)
        b = np.round(b).astype(int)

        out[:, :, 0] = r
        out[:, :, 1] = g
        out[:, :, 2] = b

        return Image.fromarray(np.transpose(out, axes=(1, 0, 2)).astype(np.uint8))

    def outImg2XYZ(self, i, j, faceIdx, faceSize):

        a = 2.0 * float(i) / faceSize
        b = 2.0 * float(j) / faceSize

        if faceIdx == 0:  # back
            (x, y, z) = (-1.0, 1.0 - a, 1.0 - b)
        elif faceIdx == 1:  # left
            (x, y, z) = (a - 1.0, -1.0, 1.0 - b)
        elif faceIdx == 2:  # front
            (x, y, z) = (1.0, a - 1.0, 1.0 - b)
        elif faceIdx == 3:  # right
            (x, y, z) = (1.0 - a, 1.0, 1.0 - b)
        elif faceIdx == 4:  # top
            (x, y, z) = (b - 1.0, a - 1.0, 1.0)
        elif faceIdx == 5:  # bottom
            (x, y, z) = (1.0 - b, a - 1.0, -1.0)
        return (x, y, z)

    def getPixelPanoramic(self, inSize, x, y, faceIdx, facesize=960):

        (x, y, z) = self.outImg2XYZ(x, y, faceIdx, facesize)
        theta = atan2(y, x)  # range -pi to pi
        r = hypot(x, y)
        phi = atan2(z, r)  # range -pi/2 to pi/2

        # source img coords
        uf = 0.5 * inSize[0] * (theta + pi) / pi
        vf = 0.5 * inSize[0] * (pi / 2 - phi) / pi

        return (uf, vf), (x,y,z)


if __name__ == '__main__':

    image = Image.open("/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/images_UHD_indoor/image_0.png")
    cube = CubeProjection(image, "/home/iaslab")


    sides = cube.cube_projection()

    # Conversion of full image for testing
    # getPixelPanoramic
    inSize = image.size
    remap = Image.new('RGB', image.size)
    in_img = image.load()
    out_img = remap.load()

    for idf in range(6):
        for id_x in range(960):
            for id_y in range(960):
                u, v = cube.getPixelPanoramic(inSize, id_x, id_y, idf)
                out_img[u, v] = in_img[u, v]

    plt.imshow(remap)
    plt.show()