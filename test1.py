from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Đường dẫn của hình ảnh gốc
input_image_path = "C:\\Users\\ducva\\Downloads\\fractal\\anime.jpg"

# Đọc hình ảnh gốc
input_image = Image.open(input_image_path)

# Đường dẫn để lưu ảnh nén
compressed_image_path = "compressed_image.jpg"

# Lưu ảnh dưới định dạng JPEG với mức nén (quality). Chất lượng từ 0 (tệ nhất) đến 100 (tốt nhất).
quality = 85
input_image.save(compressed_image_path, "JPEG", quality=quality)

# Đọc lại ảnh nén
compressed_image = Image.open(compressed_image_path)

# Hàm tính PSNR cho toàn bộ hình ảnh
def calculate_image_psnr(original, compressed):
    original = np.array(original)
    compressed = np.array(compressed)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Tính toán PSNR của hình ảnh nén
psnr_value = calculate_image_psnr(input_image, compressed_image)

# Tính toán kích thước của ảnh gốc và ảnh nén
original_image_size = os.path.getsize(input_image_path)
compressed_image_size = os.path.getsize(compressed_image_path)

# Hiển thị hình ảnh gốc và hình ảnh sau khi nén
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(input_image)
axs[0].set_title(f'Hình ảnh gốc\nKích thước: {original_image_size} bytes')

axs[1].imshow(compressed_image)
axs[1].set_title(f'Hình ảnh sau khi nén\nPhương pháp nén ảnh: JPEG\nKích thước ảnh sau khi nén: {compressed_image_size} bytes\nPSNR: {psnr_value:.2f} dB')

plt.tight_layout()
plt.show()

print(f"Kích thước ảnh gốc: {original_image_size} bytes")
print(f"Kích thước ảnh nén: {compressed_image_size} bytes")
print(f"PSNR: {psnr_value:.2f} dB")
