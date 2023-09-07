import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from PIL import ImageDraw
import os
import os.path as osp

def plot_scans(points, det=None, path="./" ,number="1", people=None):
    plt.figure(1, figsize=(8,8))
    plt.xlim(-14.0, 14.0)
    plt.ylim(-14.0, 14.0)
    # plt.xlim(-7.0, 7.0)
    # plt.ylim(-7.0, 7.0)

    plt.scatter(points[:, 0], points[:, 1], label='Points', color='black', marker='x', s=0.5)

    if det is not None:

        if people is not None:
            colors = cm.rainbow(np.linspace(0, 1, max(people)+1))
            people_colors = []
            for i in range(len(people)):
                people_colors.append(colors[people[i]])
            plt.scatter(det[:,0], det[:,1], label='Points', color=people_colors, marker='.')

        else:
            plt.scatter(det[:, 0], det[:, 1], label='Points', color='blue', marker='.')

    plt.savefig(osp.join(path, "img"+number.zfill(4)+".png"))
    plt.close()

def plot_detection(im, detection, num, path= '/media/leonardo/Elements/prova/yolo_out/'):
    pil_image = Image.fromarray(im)
    draw = ImageDraw.Draw(pil_image)

    # Default font for labels
    pp_data = []
    for d in detection:
        # Draw the bounding box
        draw.rectangle(d, outline="red", width=2)
    pil_image.save(path + str(num) + '.jpg')