import numpy as np
from scipy.linalg import pinv2

def grappa_reconstruction(kspace, kernel_size, acceleration_factor):
    num_coils, num_rows, num_cols = kspace.shape
    kspace_recon = np.zeros_like(kspace)
    
    # 针对每个像素位置进行重建
    for row in range(num_rows):
        for col in range(num_cols):
            if (row + kernel_size) <= num_rows and (col + kernel_size) <= num_cols:
                # 提取训练区域的k空间数据
                kspace_train = kspace[:, row:row+kernel_size, col:col+kernel_size]
                kspace_train = np.reshape(kspace_train, (num_coils, -1))
                
                # 计算加权矩阵
                A = np.transpose(kspace_train)
                AtA_inv = pinv2(np.matmul(np.transpose(A), A))
                
                # 选择参考线的位置（通常为中心位置）
                ref_row = row + kernel_size // 2
                ref_col = col + kernel_size // 2
                
                # 提取参考线的k空间数据
                ref_data = kspace[:, ref_row, ref_col]
                
                # 计算加权矩阵与参考线的乘积
                B = np.matmul(np.transpose(A), ref_data)
                
                # 使用加权矩阵和参考线数据进行重建
                kspace_recon[:, row, col] = np.matmul(AtA_inv, B)
    
    return kspace_recon

# 模拟输入数据
num_coils = 8              # 接收线圈数量
num_rows = 256             # k空间行数
num_cols = 256             # k空间列数
acceleration_factor = 2    # 加速因子

# 生成模拟k空间数据
kspace = np.random.randn(num_coils, num_rows, num_cols)

# 执行GRAPPA重建
recon_kspace = grappa_reconstruction(kspace, kernel_size=5, acceleration_factor=acceleration_factor)

# 输出重建后的k空间数据
print(recon_kspace.shape)  # 输出：(num_coils, num_rows, num_cols)

