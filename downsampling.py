import cv2
import numpy as np
import torch
import random

import numpy as np
import torch

def radiant_step(image, total_num, step1, step2):
    def cond1(angle, hit=freq1):
        if angle % samp_freq == hit:
            return True
        return False

    def cond2(angle, hit=freq2):
        if angle % samp_freq == hit:
            return True
        return False

    def get_line_coordinates(x0, y0, x1, y1):
        points = []
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [(x0, y0)]
        x_step = dx / steps
        y_step = dy / steps
        for i in range(steps + 1):
            x = int(x0 + i * x_step)
            y = int(y0 + i * y_step)
            points.append((x, y))
        return points + [(2*x0-x, 2*y0-y) for x, y in points[-2::-1]]

    image = torch.tensor(image, dtype=torch.complex128)
    image = image.cuda()
    center = (image.shape[1] // 2, image.shape[0] // 2)
    distance = np.sqrt(center[0] ** 2 + center[1] ** 2)

    height, width = image.shape[:2]
    sampling_points = []
    image1 = torch.zeros((height, width), dtype=torch.complex128).cuda()
    image2 = torch.zeros((height, width), dtype=torch.complex128).cuda()

    for angle in range(0, total_num, 1):
        radian = np.radians(angle/10)

        dx = int(distance * np.cos(radian))
        dy = int(distance * np.sin(radian))

        x1, y1 = center[0] + dx, center[1] + dy

        for x_coord, y_coord in line_coordinates:
            x_coord = np.clip(x_coord, 0, width - 1)
            y_coord = np.clip(y_coord, 0, height - 1)

            if cond1(angle):
                image1[x_coord, y_coord] = image[x_coord, y_coord]
            elif cond2(angle):
            #if cond2(angle, ran_num[1]):
                image2[x_coord, y_coord] = image[x_coord, y_coord]

    image1 = image1.cpu().numpy()
    image2 = image2.cpu().numpy()
    return image1, image2

def star_sampling(image, samp_freq, freq1, freq2):
    def cond1(angle, hit=freq1):
        if angle % samp_freq == hit:
            return True
        return False

    def cond2(angle, hit=freq2):
        if angle % samp_freq == hit:
            return True
        return False

    image = torch.tensor(image, dtype=torch.complex128)
    image = image.cuda()
    center = (image.shape[1] // 2, image.shape[0] // 2)
    distance = np.sqrt(center[0] ** 2 + center[1] ** 2)

    height, width = image.shape[:2]
    sampling_points = []
    angle_step = 1
    image1 = torch.zeros((height, width), dtype=torch.complex128).cuda()
    image2 = torch.zeros((height, width), dtype=torch.complex128).cuda()

    def get_line_coordinates(x0, y0, x1, y1):
        points = []
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [(x0, y0)]
        x_step = dx / steps
        y_step = dy / steps
        for i in range(steps + 1):
            x = int(x0 + i * x_step)
            y = int(y0 + i * y_step)
            points.append((x, y))
        return points + [(2*x0-x, 2*y0-y) for x, y in points[-2::-1]]

    for angle in range(0, 18000, angle_step):
        #if angle % (samp_freq * 8) == 0:
        #    ran_num = random.sample(range(0, samp_freq), 2)
        if not cond1(angle) and not cond2(angle):
            continue

        radian = np.radians(angle/100)

        dx = int(distance * np.cos(radian))
        dy = int(distance * np.sin(radian))

        x1, y1 = center[0] + dx, center[1] + dy

        line_coordinates = get_line_coordinates(center[0], center[1], x1, y1)

        for x_coord, y_coord in line_coordinates:
            x_coord = np.clip(x_coord, 0, width - 1)
            y_coord = np.clip(y_coord, 0, height - 1)

            if cond1(angle):
                image1[x_coord, y_coord] = image[x_coord, y_coord]
            if cond2(angle):
                image2[x_coord, y_coord] = image[x_coord, y_coord]
            #else:
            #    image1[x_coord, y_coord] = image[x_coord, y_coord]
            #    image2[x_coord, y_coord] = image[x_coord, y_coord]

    image1 = image1.cpu().numpy()
    image2 = image2.cpu().numpy()
    return image1, image2
