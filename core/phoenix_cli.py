import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from compressai.zoo import bmshj2018_hyperprior

class PhoenixCodec:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.models = {}
        print("Инициализация Феникса...")
        for q in [1, 3, 6]:
            self.models[q] = bmshj2018_hyperprior(quality=q, pretrained=True).to(self.device)
            self.models[q].eval()
        print("Модели загружены.")

    def process_frame(self, img_pil, quality):
        # Нейросеть ожидает квадрат 256x256
        img_tensor = to_tensor(img_pil).unsqueeze(0).to(self.device)
        model = self.models[quality]
        
        with torch.no_grad():
            out_net = model.compress(img_tensor)
            size = sum(len(b) for s in out_net["strings"] for b in s)
            out_dec = model.decompress(out_net["strings"], out_net["shape"])
            recon = torch.clamp(out_dec["x_hat"], 0, 1).squeeze(0).cpu()
            
        return to_pil_image(recon), size

def main():
    parser = argparse.ArgumentParser(description="Phoenix Codec CLI")
    parser.add_argument("--input", type=str, required=True, help="Путь к входному видео")
    parser.add_argument("--output", type=str, default="phoenix_output.mp4", help="Путь к выходному видео")
    parser.add_argument("--quality", type=int, default=6, choices=[1, 3, 6], help="Уровень качества (1=экстрим, 6=HD)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Устройство (cpu или cuda)")
    
    args = parser.parse_args()
    
    codec = PhoenixCodec(device=args.device)
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть {args.input}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- ЛОГИКА СОХРАНЕНИЯ ПРОПОРЦИЙ ---
    aspect_ratio = width / height
    target_size = 256
    
    if aspect_ratio > 1: # Широкий кадр
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else: # Высокий кадр
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
        
    top = (target_size - new_height) // 2
    bottom = target_size - new_height - top
    left = (target_size - new_width) // 2
    right = target_size - new_width - left
    
    # Настройка записи видео в исходных пропорциях
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    total_size = 0
    
    print(f"Обработка видео {width}x{height} с качеством {args.quality}...")
    print(f"Паддинг: {left}px слева, {right}px справа, {top}px сверху, {bottom}px снизу")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Ресайз с сохранением пропорций
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        # 2. Паддинг до 256x256 (черные поля)
        frame_padded = cv2.copyMakeBorder(
            frame_resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # 3. Конвертация в PIL и обработка Фениксом
        frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        recon_pil, size = codec.process_frame(img_pil, args.quality)
        total_size += size
        
        # 4. Обрезка черных полей (возврат к исходным пропорциям)
        recon_np = np.array(recon_pil)
        recon_cropped = recon_np[top:top+new_height, left:left+new_width]
        
        # 5. Ресайз обратно до исходного размера (width, height)
        recon_final = cv2.resize(recon_cropped, (width, height))
        
        # 6. Запись в видео
        recon_bgr = cv2.cvtColor(recon_final, cv2.COLOR_RGB2BGR)
        out.write(recon_bgr)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Обработано кадров: {frame_count}")
            
    cap.release()
    out.release()
    
    avg_size = total_size / frame_count if frame_count > 0 else 0
    print(f"\nГотово! Обработано {frame_count} кадров.")
    print(f"Средний размер кадра: {avg_size:.0f} байт")
    print(f"Результат сохранен в {args.output}")

if __name__ == "__main__":
    main()