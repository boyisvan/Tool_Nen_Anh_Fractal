import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

# Hàm tìm kiếm khối miền tốt nhất cho một khối phạm vi
def find_best_match(range_block, domain_blocks):
    best_block = None
    best_similarity = -1  # Đánh giá tương đồng tốt nhất

    for domain_block in domain_blocks:
        # Thay đổi kích thước khối miền về kích thước của khối phạm vi
        domain_block_resized = domain_block.resize(range_block.size, Image.LANCZOS)
        similarity = calculate_similarity(range_block, domain_block_resized)

        if similarity > best_similarity:
            best_similarity = similarity
            best_block = domain_block_resized

    return best_block, best_similarity

# Hàm tính độ tương đồng giữa hai khối sử dụng PSNR làm ví dụ
def calculate_similarity(block1, block2):
    arr1 = np.array(block1)
    arr2 = np.array(block2)

    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Hàm tính PSNR cho toàn bộ hình ảnh
def calculate_image_psnr(original, reconstructed):
    original = np.array(original)
    reconstructed = np.array(reconstructed)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Hàm áp dụng phép biến đổi affine để tái tạo lại khối phạm vi
def apply_transformation(range_block, domain_block, matrix):
    domain_array = np.array(domain_block)

    transformed_array = cv2.warpAffine(domain_array, matrix, (range_block.size[0], range_block.size[1]))

    transformed_block = Image.fromarray(transformed_array)

    return transformed_block

# Tạo một ma trận biến đổi affine
def calculate_affine_matrix(range_block, domain_block):
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    return matrix

# Kích thước của các khối phạm vi và khối miền
block_size = 32  # Kích thước của mỗi khối
domain_size = 64  # Kích thước của mỗi khối miền


# Đường dẫn của hình ảnh gốc
input_image_path = "C:\\Users\\ducva\\Downloads\\fractal\\anime.jpg"


# Load hình ảnh gốc
input_image = Image.open(input_image_path)
width, height = input_image.size

# Số lượng khối theo chiều ngang và chiều dọc
block_size = 64
num_blocks_x = width // block_size
num_blocks_y = height // block_size

# Tạo danh sách các khối phạm vi
range_blocks = []

for i in range(num_blocks_x):
    for j in range(num_blocks_y):
        range_block = input_image.crop((i * block_size, j * block_size,
                                        (i + 1) * block_size, (j + 1) * block_size))
        range_blocks.append(range_block)

# Tạo danh sách các khối miền
domain_blocks = []

for i in range(0, width - domain_size + 1, block_size):
    for j in range(0, height - domain_size + 1, block_size):
        domain_block = input_image.crop((i, j, i + domain_size, j + domain_size))
        domain_blocks.append(domain_block)

# Tìm kiếm khối miền tốt nhất cho từng khối phạm vi và lưu trữ thông số biến đổi
transformations = []

for i in range(num_blocks_x):
    for j in range(num_blocks_y):
        range_block = range_blocks[i * num_blocks_y + j]
        best_domain_block, similarity = find_best_match(range_block, domain_blocks)

        matrix = calculate_affine_matrix(range_block, best_domain_block)

        best_domain_block_array = np.array(best_domain_block).tolist()

        transformations.append({
            'range_block': (i, j),
            'domain_block': best_domain_block_array,
            'similarity': similarity,
            'matrix': matrix.tolist()
        })

# Lưu trữ thông số biến đổi vào một tệp JSON
output_file = "transformations.json"

with open(output_file, 'w') as f:
    json.dump(transformations, f, indent=4)

print(f"Lưu trữ thông số biến đổi vào tệp: {output_file}")

# Đọc các thông số biến đổi từ tệp JSON
with open(output_file, 'r') as f:
    transformations_loaded = json.load(f)

# Tạo một hình ảnh trống để lắp ráp lại
output_image = Image.new('RGB', (width, height))

# Áp dụng các phép biến đổi affine để tái tạo lại hình ảnh
for transformation in transformations_loaded:
    i, j = transformation['range_block']
    domain_block_array = transformation['domain_block']
    matrix = np.array(transformation['matrix'], dtype=np.float32)

    domain_block = Image.fromarray(np.array(domain_block_array, dtype=np.uint8))

    reconstructed_block = apply_transformation(range_blocks[i * num_blocks_y + j], domain_block, matrix)

    output_image.paste(reconstructed_block, (i * block_size, j * block_size))

# Tính toán PSNR của hình ảnh tái tạo
psnr_value = calculate_image_psnr(input_image, output_image)

# Lưu ảnh tái tạo lại dưới dạng JPEG
output_image_path = "output_image.jpg"
output_image.save(output_image_path, "JPEG")

# Tính toán kích thước của ảnh tái tạo lại
reconstructed_image_size = os.path.getsize(output_image_path)

# Hiển thị hình ảnh gốc và hình ảnh sau khi tái tạo
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(input_image)
axs[0].set_title(f'Hình ảnh gốc\nKích thước: {os.path.getsize(input_image_path)} bytes')

axs[1].imshow(output_image)
axs[1].set_title(f'Hình ảnh sau khi tái tạo\nPhương pháp nén ảnh : fractal\nKích thước ảnh sau khi tái tạo: {reconstructed_image_size} bytes\nPSNR: {psnr_value:.2f} dB')

plt.tight_layout()
plt.show()

print(f"Kích thước ảnh tái tạo lại: {reconstructed_image_size} bytes")
