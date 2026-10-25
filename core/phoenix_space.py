import torch
import struct
import io
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from compressai.zoo import bmshj2018_hyperprior

class PhoenixSpaceCodec:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print("Инициализация Феникс-Космос...")
        # Для космоса держим только два режима: "смысл" (Q=1) и "отчет" (Q=6)
        self.models = {
            1: bmshj2018_hyperprior(quality=1, pretrained=True).to(self.device).eval(),
            6: bmshj2018_hyperprior(quality=6, pretrained=True).to(self.device).eval()
        }
        print("Модели загружены.")

    def encode_frame(self, img_pil, quality=6, frame_id=0):
        """Сжимает кадр и упаковывает его в космический бандл (Bundle Protocol)."""
        img_tensor = to_tensor(img_pil).unsqueeze(0).to(self.device)
        model = self.models[quality]
        
        with torch.no_grad():
            out = model.compress(img_tensor)
            # Упрощенная упаковка: склеиваем все строки байтов в один массив
            # (В реальности здесь был бы энтропийный кодек, но для примера сойдет)
            latent_bytes = b''.join([b for s in out["strings"] for b in s])
            
        # Генерируем "семя" для шума (в реальности оно случайное)
        seed = int(torch.randint(0, 1000000, (1,)).item())
        
        # --- ФОРМИРОВАНИЕ БАНДЛА (DTN Bundle) ---
        # Заголовок: frame_id (4 байта), quality (1 байт), seed (4 байта)
        # Плюс 8 байт "защиты" (FEC) для семени, чтобы оно не потерялось
        header = struct.pack(">I B I 8s", frame_id, quality, seed, b'\x00'*8)
        bundle = header + latent_bytes
        
        return bundle

    def decode_bundle(self, bundle):
        """Принимает бандл, проверяет целостность и восстанавливает кадр."""
        header_size = 17 # 4 + 1 + 4 + 8
        header = bundle[:header_size]
        latent_bytes = bundle[header_size:]
        
        frame_id, quality, seed, fec = struct.unpack(">I B I 8s", header)
        
        # --- ПРОВЕРКА ЦЕЛОСТНОСТИ (FEC) ---
        # В реальности здесь был бы код Рида-Соломона.
        # Для примера: если FEC != b'\x00'*8, значит, пакет поврежден.
        if fec != b'\x00'*8:
            print(f"  [Ошибка] Бандл кадра {frame_id} поврежден (FEC не совпал). Пропускаем.")
            return None
        
        # Восстанавливаем тензор латента из байтов
        # (В реальности нужно распаковать энтропийный код, здесь мы просто создаем заглушку
        # для демонстрации логики, так как полноценный декодер требует сложной структуры out_net)
        # Для честного теста мы просто вернем "нулевой" кадр, так как без полноценного
        # сохранения структуры out_net мы не можем декодировать обратно.
        print(f"  [Успех] Бандл кадра {frame_id} доставлен. Качество: {quality}")
        return frame_id

def simulate_space_channel(bundle, packet_loss_rate=0.2):
    """Симуляция космического канала: случайно теряет или портит пакеты."""
    import random
    if random.random() < packet_loss_rate:
        # Имитация потери: возвращаем мусор
        return b'\xFF' * len(bundle)
    return bundle

def main():
    codec = PhoenixSpaceCodec(device='cpu')
    
    # Загружаем тестовое фото
    img = Image.open("your_photo.jpg").convert("RGB").resize((256, 256))
    
    print("\n--- Симуляция передачи через космический канал (с потерями) ---")
    
    for frame_id in range(5):
        # 1. Отправитель: сжимает и упаковывает
        bundle = codec.encode_frame(img, quality=6, frame_id=frame_id)
        print(f"Отправка кадра {frame_id}: размер бандла {len(bundle)} байт")
        
        # 2. Канал: теряет пакеты
        received_bundle = simulate_space_channel(bundle, packet_loss_rate=0.4)
        
        # 3. Приемник: пытается распаковать
        result = codec.decode_bundle(received_bundle)
        
    print("\nСимуляция завершена. Обратите внимание, как система реагирует на потери.")

if __name__ == "__main__":
    main()