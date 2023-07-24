import cv2
import numpy as np
import torch

def is_central(cthred, x, y, width, height):
    if ((x > width//2 - cthred and x < width//2 + cthred) 
    and (y > height//2 - cthred and y < height//2 + cthred)):
        return True
    else:
        return False

def samp(fft):
    height, width = fft.shape[:2]

    image1 = np.zeros((height, width), dtype=np.complex128)
    image2 = np.zeros((height, width), dtype=np.complex128)

    cthred = (width//2)*(1/10)

    for x in range(width):
        for y in range(height):
            if x % 2 == 0 or is_central(cthred, x, y, width, height):
            #if x % 2 == 0:
                image1[x, y] = fft[x, y]
            if x % 2 == 1 or is_central(cthred, x, y, width, height):
            #if x % 2 == 1:
                image2[x, y] = fft[x, y]

    return image1, image2

def spiral_sampling(image, clockwise=True):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    num_points = width//4
    image1 = np.zeros((height, width), dtype=np.complex128)
    image2 = np.zeros((height, width), dtype=np.complex128)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    angle_step = 1

    for angle in range(0, 360, angle_step):
        radian = np.radians(angle)
        angles = np.linspace(angle, 2 * np.pi + angle, num_points, endpoint=False)
        distances = np.linspace(0, max_dist, num_points)

        if not clockwise:
            angles = np.flipud(angles)
        x = center_x + distances * np.cos(angles)
        y = center_y + distances * np.sin(angles)
        for p in np.column_stack((x, y)):
            x = int(p[0])
            y = int(p[1])
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            if angle % 2 == 0:
                image1[x, y] = image[x, y]
            if angle % 2 == 1:
                image2[x, y] = image[x, y]

    return image1, image2

def horiz_samp_four(fft, centra_size):
    height, width = fft.shape[:2]
    cthred = (width//2)*centra_size

    image1 = np.zeros((height, width), dtype=np.complex128)
    image2 = np.zeros((height, width), dtype=np.complex128)
    image3 = np.zeros((height, width), dtype=np.complex128)
    image4 = np.zeros((height, width), dtype=np.complex128)

    for x in range(width):
        for y in range(height):
            if x % 4 == 0 or is_central(cthred, x, y, width, height):
            #if x % 4 == 0:
                image1[x, y] = fft[x, y]
            if x % 4 == 1 or is_central(cthred, x, y, width, height):
            #if x % 4 == 1:
                image2[x, y] = fft[x, y]
            if x % 4 == 2 or is_central(cthred, x, y, width, height):
            #if x % 4 == 2:
                image3[x, y] = fft[x, y]
            if x % 4 == 3 or is_central(cthred, x, y, width, height):
            #if x % 4 == 3:
                image4[x, y] = fft[x, y]

    return image1, image2, image3, image4

#Bresenham
def star_sampling(image, central_freq, samp_freq, centre_size):
    image = torch.tensor(image, dtype=torch.complex128) 
    image = image.cuda()
    center = (image.shape[1] // 2, image.shape[0] // 2)
    #distance = min(image.shape[:2]) // 2
    distance = np.sqrt(center[0]**2 + center[1]**2)

    height, width = image.shape[:2]
    sampling_points = []
    angle_step = 1 
    image1 = torch.zeros((height, width), dtype=torch.complex128).cuda() 
    image2 = torch.zeros((height, width), dtype=torch.complex128).cuda()
    cthred = (width//2)*centre_size

    """
    for x in range(width):
        for y in range(height):
            if is_central(cthred, x, y, width, height):
                image1[x, y] = image[x, y]
                image2[x, y] = image[x, y]
    """

    for angle in range(0, 36000, angle_step):
        radian = np.radians(angle/100)

        dx = int(distance * np.cos(radian))
        dy = int(distance * np.sin(radian))

        x0, y0 = center[0], center[1]
        x1, y1 = center[0] + dx, center[1] + dy
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        deltax = x1 - x0
        deltay = abs(y1 - y0)
        error = int(deltax / 2)
        y = y0
        ystep = -1 if y0 > y1 else 1
        for x in range(x0, x1 + 1):
            x_coord = y if steep else x
            y_coord = x if steep else y

            x_coord = np.clip(x_coord, 0, width - 1)
            y_coord = np.clip(y_coord, 0, height - 1)

            if is_central(cthred, x, y, width, height):
                if angle % central_freq == 0:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                elif angle % central_freq == (central_freq//2 - 1):
                    image2[x_coord, y_coord] = image[x_coord, y_coord]
                else:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                    image2[x_coord, y_coord] = image[x_coord, y_coord]
            else:
                if angle % samp_freq == 0:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                elif angle % samp_freq == (samp_freq//2 - 1):
                    image2[x_coord, y_coord] = image[x_coord, y_coord]
                else:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                    image2[x_coord, y_coord] = image[x_coord, y_coord]

            error -= deltay
            if error < 0:
                y += ystep
                error += deltax

    image1 = image1.cpu().numpy()
    image2 = image2.cpu().numpy()
    #print(image1, image2)
    return image1, image2
