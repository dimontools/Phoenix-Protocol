import torch
import numpy as np
import io
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from compressai.zoo import bmshj2018_hyperprior

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    print("=== Phoenix Codec: Verification Script ===")
    print("Сравнение: Феникс (Quality=6) vs JPEG (Quality=90) vs JPEG (Quality=20)\n")
    
    # 1. Загружаем тестовое изображение
    try:
        img = Image.open("test_image.jpg").convert("RGB").resize((256, 256))
        print("Используется файл test_image.jpg")
    except FileNotFoundError:
        print("Файл test_image.jpg не найден. Генерирую случайное изображение (для теста).")
        # Генерируем "лицо" из градиентов, чтобы тест был осмысленным
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                arr[i, j] = [i, j, (i+j)//2]
        img = Image.fromarray(arr)
    
    tensor = to_tensor(img).unsqueeze(0)
    
    # --- 2. ТЕСТ ФЕНИКСА ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bmshj2018_hyperprior(quality=6, pretrained=True).to(device).eval()
    tensor = tensor.to(device)
    
    with torch.no_grad():
        out_net = model.compress(tensor)
        size_phoenix = sum(len(b) for s in out_net["strings"] for b in s)
        out_dec = model.decompress(out_net["strings"], out_net["shape"])
        recon_phoenix = torch.clamp(out_dec["x_hat"], 0, 1)
    
    psnr_phoenix = calculate_psnr(tensor, recon_phoenix).item()
    
    # --- 3. ТЕСТ JPEG-90 ---
    buffer_90 = io.BytesIO()
    img.save(buffer_90, format="JPEG", quality=90)
    size_jpeg90 = len(buffer_90.getvalue())
    buffer_90.seek(0)
    jpeg90_img = Image.open(buffer_90).convert("RGB")
    jpeg90_tensor = to_tensor(jpeg90_img).unsqueeze(0).to(device)
    psnr_jpeg90 = calculate_psnr(tensor, jpeg90_tensor).item()
    
    # --- 4. ТЕСТ JPEG-20 ---
    buffer_20 = io.BytesIO()
    img.save(buffer_20, format="JPEG", quality=20)
    size_jpeg20 = len(buffer_20.getvalue())
    buffer_20.seek(0)
    jpeg20_img = Image.open(buffer_20).convert("RGB")
    jpeg20_tensor = to_tensor(jpeg20_img).unsqueeze(0).to(device)
    psnr_jpeg20 = calculate_psnr(tensor, jpeg20_tensor).item()
    
    # --- 5. ВЫВОД РЕЗУЛЬТАТОВ ---
    print(f"{'Метод':<20} | {'Размер (байт)':<15} | {'PSNR (dB)':<10}")
    print("-" * 50)
    print(f"{'Феникс (Quality=6)':<20} | {size_phoenix:<15} | {psnr_phoenix:<10.2f}")
    print(f"{'JPEG (Quality=90)':<20} | {size_jpeg90:<15} | {psnr_jpeg90:<10.2f}")
    print(f"{'JPEG (Quality=20)':<20} | {size_jpeg20:<15} | {psnr_jpeg20:<10.2f}")
    print("-" * 50)
    print("\nВывод:")
    print(f"Феникс в {size_jpeg90 / size_phoenix:.1f} раз меньше JPEG-90.")
    print(f"При этом PSNR у Феникса {'выше' if psnr_phoenix > psnr_jpeg20 else 'ниже'}, чем у JPEG-20.")

if __name__ == "__main__":
    main()
