import numpy as np
import cv2
from matplotlib import pyplot as plt

# 逆运算步骤1: 读取保存的图像文件
res1 = cv2.imread("./out_norm.tif", 0)
# Perform inverse DFT
idft = cv2.idft(np.exp(1j * np.angle(res1)), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
idft = np.uint8(idft)

# Save the reversed image
cv2.imwrite("out_reverse.tif", idft)
