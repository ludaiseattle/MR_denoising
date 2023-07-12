import cv2
import numpy as np

def is_central(cthred, x, y, width, height):
    if ((x > width//2 - cthred and x < width//2 + cthred) 
    and (y > height//2 - cthred and y < height//2 + cthred)):
        return True
    else:
        return False

#Bresenham
def star_sampling(image):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    #distance = min(image.shape[:2]) // 2
    distance = np.sqrt(center[0]**2 + center[1]**2)

    height, width = image.shape[:2]
    sampling_points = []
    angle_step = 1 
    image1 = np.zeros((height, width), dtype=np.complex128)
    image2 = np.zeros((height, width), dtype=np.complex128)
    cthred = (width//2)*(1/2)

    for angle in range(0, 360, angle_step):
        radian = np.radians(angle)

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
                if angle % 16 == 0:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                elif angle % 16 == 4:
                    image2[x_coord, y_coord] = image[x_coord, y_coord]
                else:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                    image2[x_coord, y_coord] = image[x_coord, y_coord]
            else:
                if angle % 8 == 0:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                elif angle % 8 == 4:
                    image2[x_coord, y_coord] = image[x_coord, y_coord]
                else:
                    image1[x_coord, y_coord] = image[x_coord, y_coord]
                    image2[x_coord, y_coord] = image[x_coord, y_coord]

            error -= deltay
            if error < 0:
                y += ystep
                error += deltax

    print(image1, image2)
    return image1, image2
