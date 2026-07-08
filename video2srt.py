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
    batch_size=32, 
    word_timestamps=True,          # 必須開啟文字時間戳
    max_new_tokens=None,
    # 限制每個 segment 的字數，中文字建議設在 20 ~ 25 左右
    condition_on_previous_text=False,
    initial_prompt="這是一堂關於GitHub實務操作的電機系AGILab實驗室的訓練課程，會提到Git push等術語。"
)

print(f"Detected language: {info.language} with probability {info.language_probability}")
#%%
# 輸出為 SRT 檔案
#%%
# 輸出為 SRT 檔案 (專為 AGILab 課程優化的自動換行、限制雙行版)
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

MAX_CHARS_PER_LINE = 20  # 單行最多 20 個中文字

with open("output.srt", "w", encoding="utf-8") as f:
    srt_idx = 1
    
    for segment in segments:
        text = segment.text.strip().replace(" ", "") # 移除中文間可能產生的空格
        
        # 如果整段字數很少，直接寫入，不需要折行
        if len(text) <= MAX_CHARS_PER_LINE:
            f.write(f"{srt_idx}\n")
            f.write(f"{format_time(segment.start)} --> {format_time(segment.end)}\n")
            f.write(f"{text}\n\n")
            srt_idx += 1
            continue
            
        # 如果字數多於單行限制，我們利用 words 資訊，將其拆分成多個「最多兩行」的獨立字幕塊
        current_words = []
        current_len = 0
        
        # 遍歷這個片段裡面的每一個字
        words_list = list(segment.words) if segment.words else []
        
        if not words_list:
            # 萬一沒有字時間戳的備用方案：直接硬切兩行
            f.write(f"{srt_idx}\n")
            f.write(f"{format_time(segment.start)} --> {format_time(segment.end)}\n")
            line1 = text[:MAX_CHARS_PER_LINE]
            line2 = text[MAX_CHARS_PER_LINE:MAX_CHARS_PER_LINE*2] # 最多吃 40 字
            f.write(f"{line1}\n{line2}\n\n" if line2 else f"{line1}\n\n")
            srt_idx += 1
            continue

        # 有精確字時間戳的情況（標準狀況）：
        sub_chunk_words = []
        for word in words_list:
            sub_chunk_words.append(word)
            # 累積字數達到 40 字（剛好夠排滿 2 行），或者已經到片段結尾時就輸出一塊字幕
            current_text = "".join([w.word for w in sub_chunk_words]).strip().replace(" ", "")
            
            if len(current_text) >= MAX_CHARS_PER_LINE * 2:
                # 切成兩行
                line1 = current_text[:MAX_CHARS_PER_LINE]
                line2 = current_text[MAX_CHARS_PER_LINE:]
                
                f.write(f"{srt_idx}\n")
                f.write(f"{format_time(sub_chunk_words[0].start)} --> {format_time(sub_chunk_words[-1].end)}\n")
                f.write(f"{line1}\n{line2}\n\n")
                srt_idx += 1
                sub_chunk_words = [] # 清空，準備下一塊
                
        # 處理尾巴剩下不滿 40 字的部分
        if sub_chunk_words:
            final_text = "".join([w.word for w in sub_chunk_words]).strip().replace(" ", "")
            if len(final_text) > MAX_CHARS_PER_LINE:
                line1 = final_text[:MAX_CHARS_PER_LINE]
                line2 = final_text[MAX_CHARS_PER_LINE:]
                final_output_text = f"{line1}\n{line2}"
            else:
                final_output_text = final_text
                
            f.write(f"{srt_idx}\n")
            f.write(f"{format_time(sub_chunk_words[0].start)} --> {format_time(segment.end)}\n")
            f.write(f"{final_output_text}\n\n")
            srt_idx += 1

print("SRT 檔案已成功生成完美雙行版: output.srt")
# %%
# %%
