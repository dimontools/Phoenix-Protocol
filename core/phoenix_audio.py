import torch
import torchaudio
import soundfile as sf
import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio

# 1. Загружаем модель EnCodec
print("Загрузка аудио-кодека...")
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(1.5)  # 1.5 kbps — экстремально низкий битрейт
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Загружаем аудиофайл через soundfile
audio_path = "test_audio.wav"
print(f"Загрузка {audio_path}...")

# Читаем файл
data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
# soundfile возвращает [time, channels], нам нужно [channels, time]
wav = torch.from_numpy(data.T).contiguous()

# Приводим к нужному формату (24 кГц, моно)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0).to(device)  # Добавляем batch dimension

# 3. Сжатие (в латент)
print("Сжатие...")
with torch.no_grad():
    encoded_frames = model.encode(wav)
    
    # Считаем размер сжатых данных
    total_bytes = 0
    for frame in encoded_frames:
        # frame[0] — это тензор с кодами, frame[1] — скейл (если есть)
        total_bytes += frame[0].numel() * 2  # 2 байта на код (int16)
    print(f"Размер сжатого файла: {total_bytes} байт")
    
    # Считаем длительность в секундах, чтобы понять битрейт
    duration = wav.shape[-1] / model.sample_rate
    print(f"Длительность: {duration:.2f} сек")
    print(f"Реальный битрейт: {total_bytes * 8 / duration / 1000:.1f} kbps")

# 4. Восстановление (декодирование)
print("Восстановление...")
with torch.no_grad():
    decoded_wav = model.decode(encoded_frames)

# 5. Сохранение результата
output_path = "test_audio_phoenix.wav"
# torchaudio.save тоже может требовать torchcodec, поэтому используем soundfile для сохранения
decoded_np = decoded_wav.squeeze(0).cpu().numpy().T  # [time, channels]
sf.write(output_path, decoded_np, model.sample_rate)
print(f"Готово! Результат сохранен в {output_path}")