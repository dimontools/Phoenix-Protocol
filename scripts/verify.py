import torch
import numpy as np
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
    
    # 1. Загружаем тестовое изображение (можно заменить на любое)
    # Для теста используем случайный шум, если нет файла
    try:
        img = Image.open("test_image.jpg").convert("RGB").resize((256, 256))
    except FileNotFoundError:
        print("Файл test_image.jpg не найден. Использую случайный тензор для теста.")
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    tensor = to_tensor(img).unsqueeze(0)
    
    # 2. Загружаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bmshj2018_hyperprior(quality=6, pretrained=True).to(device).eval()
    tensor = tensor.to(device)
    
    # 3. Сжатие
    with torch.no_grad():
        out_net = model.compress(tensor)
        size = sum(len(b) for s in out_net["strings"] for b in s)
        
        # 4. Декодирование
        out_dec = model.decompress(out_net["strings"], out_net["shape"])
        recon = torch.clamp(out_dec["x_hat"], 0, 1)
    
    # 5. Метрики
    psnr = calculate_psnr(tensor, recon).item()
    
    print(f"Размер сжатого файла: {size} байт")
    print(f"PSNR: {psnr:.2f} dB")
    print("(Чем выше PSNR, тем лучше. Идеал ~40+ dB)")

if __name__ == "__main__":
    main()
