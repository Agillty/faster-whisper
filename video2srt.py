#%%



#%%
import os
import sys

if os.name == 'nt':
    conda_bin_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin')
    cudnn_bin_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')
    
    # 這是最關鍵的一步：使用 os.add_dll_directory 並獲取其 handle
    # 這能確保 Python 執行期環境強制從這裡載入，無視系統快取
    try:
        os.add_dll_directory(conda_bin_path)
        os.add_dll_directory(cudnn_bin_path)
        print("DLL 路徑強制注入成功")
    except Exception as e:
        print(f"路徑注入警告: {e}")

# 確保在匯入 Faster-Whisper 前，環境變數 PATH 已經包含該路徑
os.environ["PATH"] = conda_bin_path + os.pathsep + cudnn_bin_path + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel,BatchedInferencePipeline

model_size = "large-v3"
# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")



#%%
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe("audio.mp3", beam_size=5)


# segments, info = model.transcribe("2026_Github01_raw1.mp4", 
#     beam_size=5,
#     initial_prompt="這是一堂關於GitHub實務操作的電機系AGILab實驗室的訓練課程，會提到Git push等術語。")


batched_model = BatchedInferencePipeline(model=model)

# 3. 改用 batched_model.transcribe，現在它就支援 batch_size 了
segments, info = batched_model.transcribe(
    "2026_Github01_raw1.mp4", 
    beam_size=5,
    batch_size=8, 
    initial_prompt="這是一堂關於GitHub實務操作的電機系AGILab實驗室的訓練課程，會提到Git push等術語。"
)

print(f"Detected language: {info.language} with probability {info.language_probability}")
#%%
# 輸出為 SRT 檔案
with open("output.srt", "w", encoding="utf-8") as f:
    for i, segment in enumerate(segments):
        # 將秒數轉換為 SRT 格式的時間碼 (HH:MM:SS,mmm)
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

        f.write(f"{i+1}\n")
        f.write(f"{format_time(segment.start)} --> {format_time(segment.end)}\n")
        f.write(f"{segment.text.strip()}\n\n")

print("SRT 檔案已成功生成: output.srt")
# %%
