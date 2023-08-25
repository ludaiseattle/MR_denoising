import torch
import numpy as np
import random

def star_sampling(image, factor, samp1_num, angle_start1, samp2_num, angle_start2):
    total_num = 1800 * factor
    angle_interval1 = int(total_num/samp1_num)
    angle_interval2 = int(total_num/samp2_num)
    image = torch.tensor(image, dtype=torch.complex128)
    image = image.cuda()
    center = (image.shape[1] // 2, image.shape[0] // 2)
    #distance = np.sqrt(center[0] ** 2 + center[1] ** 2)
    distance = center[0]

    height, width = image.shape[:2]
    sampling_points = []
    angle_step = 1

    image1_mask = torch.zeros((height, width), dtype=torch.bool).cuda()
    image2_mask = torch.zeros((height, width), dtype=torch.bool).cuda()

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

    #hit1 = random.randint(0, angle_interval1)
    #hit2 = random.randint(0, angle_interval2)
    start1 = 0
    end1 = angle_interval1
    nums1 = []
    for i in range(0, samp1_num): 
        hit1 = random.randint(start1, end1)
        nums1.append(hit1)
        start1 = start1 + angle_interval1
        end1 = end1 + angle_interval1
         
    start2 = 0
    end2 = angle_interval2
    nums2 = []
    for i in range(0, samp2_num): 
        hit2 = random.randint(start2, end2)
        nums2.append(hit2)
        start2 = start2 + angle_interval2
        end2 = end2 + angle_interval2

    for angle in range(0, total_num * factor, angle_step):
        if angle == 450:
            cond1 = True
        elif angle in nums1:
            cond1 = True
        else:
            cond1 = False

        if angle == 450:
            cond2 = True
        elif angle in nums2:
            cond2 = True
        else:
            cond2 = False

        if not cond1 and not cond2:
            continue

        radian = np.radians(angle / (10 * factor))

        dx = int(distance * np.cos(radian))
        dy = int(distance * np.sin(radian))

        x1, y1 = center[0] + dx, center[1] + dy

        line_coordinates = get_line_coordinates(center[0], center[1], x1, y1)

        for x_coord, y_coord in line_coordinates:
            x_coord = np.clip(x_coord, 0, width - 1)
            y_coord = np.clip(y_coord, 0, height - 1)

            if cond1:
                image1_mask[x_coord, y_coord] = True
            if cond2:
                image2_mask[x_coord, y_coord] = True

    image1 = torch.where(image1_mask, image, torch.zeros_like(image))
    image2 = torch.where(image2_mask, image, torch.zeros_like(image))

    image1 = image1.cpu().numpy()
    image2 = image2.cpu().numpy()
    return image1, image2, image1_mask.cpu().numpy(), image2_mask.cpu().numpy()
