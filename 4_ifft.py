import numpy as np
from PIL import Image

def inverse_fourier_transform(kspace):
    image = np.fft.ifftshift(kspace)  # 反移中心
    image = np.fft.ifft2(image)      # 反二维傅里叶变换
    image = np.abs(image)            # 取绝对值
    image = np.uint8(image)          # 转换为8位无符号整数类型
    return image

# 读取k空间数据
kspace_path = "./reconstructed_samp1_out_norm.tif"
kspace_image = Image.open(kspace_path)
kspace = np.array(kspace_image)

# 进行逆傅里叶变换
reconstructed_image = inverse_fourier_transform(kspace)

# 保存为TIFF文件
output_path = 'ifft_image.tif'
reconstructed_image_pil = Image.fromarray(reconstructed_image)
reconstructed_image_pil.save(output_path)

