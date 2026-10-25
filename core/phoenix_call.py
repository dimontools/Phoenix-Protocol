import cv2
import torch
import numpy as np
import sounddevice as sd
import threading
import queue
import time
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from compressai.zoo import bmshj2018_hyperprior
from encodec import EncodecModel
from encodec.utils import convert_audio

# --- НАСТРОЙКИ ---
VIDEO_SIZE = (256, 256)
AUDIO_SR = 24000
AUDIO_CHUNK = 2400
VIDEO_QUALITY = 6
AUDIO_BANDWIDTH = 1.5
MIC_DEVICE = 17
SPEAKER_DEVICE = 16

# --- ИНИЦИАЛИЗАЦИЯ ---
print("Загрузка моделей...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_model = bmshj2018_hyperprior(quality=VIDEO_QUALITY, pretrained=True).to(device).eval()

audio_model = EncodecModel.encodec_model_24khz()
audio_model.set_target_bandwidth(AUDIO_BANDWIDTH)
audio_model.eval()
audio_model.to(device)

print("Модели загружены.")

# --- ОЧЕРЕДИ ---
video_queue = queue.Queue(maxsize=1)
audio_play_queue = queue.Queue()

# --- ПОТОК ВИДЕО ---
def video_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть веб-камеру.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb).resize(VIDEO_SIZE)
        img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out_net = video_model.compress(img_tensor)
            out_dec = video_model.decompress(out_net["strings"], out_net["shape"])
            recon = torch.clamp(out_dec["x_hat"], 0, 1).squeeze(0).cpu()
        recon_np = (recon.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)
        if not video_queue.empty():
            try:
                video_queue.get_nowait()
            except queue.Empty:
                pass
        video_queue.put(recon_bgr)
    cap.release()

# --- ПОТОК МИКРОФОНА (Захват + Обработка) ---
def mic_thread():
    audio_buffer = np.zeros((AUDIO_CHUNK, 2), dtype=np.float32)
    
    def callback(indata, frames, time_info, status):
        nonlocal audio_buffer
        if status:
            print(status)
        audio_buffer = np.vstack((audio_buffer, indata))
        audio_buffer = audio_buffer[-AUDIO_CHUNK:]
        
        if len(audio_buffer) >= AUDIO_CHUNK:
            chunk_data = np.mean(audio_buffer, axis=1, keepdims=True)
            wav = torch.from_numpy(chunk_data.T).contiguous()
            wav = convert_audio(wav, 44100, audio_model.sample_rate, audio_model.channels)
            wav = wav.unsqueeze(0).to(device)
            with torch.no_grad():
                encoded_frames = audio_model.encode(wav)
                decoded_wav = audio_model.decode(encoded_frames)
            decoded_np = decoded_wav.squeeze(0).cpu().numpy().T
            
            # Ресемплим в 44100 для динамиков
            decoded_tensor = torch.from_numpy(decoded_np.T).contiguous()
            decoded_tensor = convert_audio(decoded_tensor, audio_model.sample_rate, 44100, 1)
            final_audio = decoded_tensor.squeeze(0).numpy().T
            
            # --- ФИКС: Ограничиваем очередь, чтобы не копить задержку ---
            # Если очередь уже имеет 5 кусков, удаляем самый старый
            while audio_play_queue.qsize() >= 5:
                try:
                    audio_play_queue.get_nowait()
                except queue.Empty:
                    pass
            audio_play_queue.put(final_audio)
    
    try:
        with sd.InputStream(device=MIC_DEVICE, callback=callback, samplerate=44100, channels=2, dtype='float32', blocksize=1024):
            print("Микрофон открыт.")
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"Ошибка микрофона: {e}")

# --- ПОТОК ДИНАМИКОВ (Воспроизведение) ---
def speaker_thread():
    play_buffer = np.zeros((0, 1), dtype=np.float32)
    
    def output_callback(outdata, frames, time_info, status):
        nonlocal play_buffer
        if status:
            print(status)
        
        # --- ФИКС: Если буфер разросся (эхо/задержка), сбрасываем его ---
        # 44100 * 1.5 = 66150 сэмплов (примерно 1.5 секунды)
        if len(play_buffer) > 66150:
            # print("Сброс буфера (переполнение)")
            play_buffer = np.zeros((0, 1), dtype=np.float32)
        
        # Подкачиваем данные из очереди
        while len(play_buffer) < frames:
            if not audio_play_queue.empty():
                new_chunk = audio_play_queue.get()
                if new_chunk.ndim == 1:
                    new_chunk = new_chunk.reshape(-1, 1)
                play_buffer = np.vstack((play_buffer, new_chunk))
            else:
                silence = np.zeros((frames - len(play_buffer), 1), dtype=np.float32)
                play_buffer = np.vstack((play_buffer, silence))
                break
        
        # Отдаем данные
        outdata[:] = play_buffer[:frames]
        play_buffer = play_buffer[frames:]
    
    try:
        with sd.OutputStream(device=SPEAKER_DEVICE, callback=output_callback, samplerate=44100, channels=1, dtype='float32', blocksize=1024):
            print("Динамики открыты.")
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"Ошибка динамиков: {e}")

# --- ОСНОВНОЙ ЦИКЛ ---
def display_loop():
    print("Нажмите 'q' в окне видео, чтобы выйти.")
    while True:
        if not video_queue.empty():
            frame = video_queue.get()
            cv2.imshow("Phoenix Video Call (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# --- ЗАПУСК ---
video_t = threading.Thread(target=video_thread, daemon=True)
mic_t = threading.Thread(target=mic_thread, daemon=True)
speaker_t = threading.Thread(target=speaker_thread, daemon=True)

video_t.start()
mic_t.start()
speaker_t.start()

display_loop()