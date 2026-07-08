import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # 💡 停用動態編譯，改走標準 CUDA 推論

# ========================================================
# 💡 針對 transformers 4.45.0 + 新版 bitsandbytes 的相容性猴子補丁
# ========================================================
import transformers.integrations.bitsandbytes as bnb_integration

# 原本的函式會因為 frozenset.discard() 而報錯，我們寫一個修正版
def patched_validate_bnb_multi_backend_availability(raise_exception=False):
    import bitsandbytes as bnb
    if not hasattr(bnb, "get_available_devices"):
        return False
    try:
        # 將不可變的 frozenset 轉為標準可變的 set 集合，避免 discard 報錯
        available_devices = set(bnb.get_available_devices())
        available_devices.discard("cpu")
        return len(available_devices) > 0
    except Exception:
        if raise_exception:
            raise
        return False

# 強制覆蓋 transformers 內部的錯誤實作
bnb_integration._validate_bnb_multi_backend_availability = patched_validate_bnb_multi_backend_availability
# ========================================================


import time  # 💡 記得在檔案最上方 import time
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 3090 Ti 環境優化與 DLL 強制注入 (Windows 必備)
# ==========================================
if os.name == 'nt':
    conda_bin_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin')
    cudnn_bin_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')
    try:
        os.add_dll_directory(conda_bin_path)
        os.add_dll_directory(cudnn_bin_path)
        print("💡 [環境] CUDA/cuDNN DLL 執行期路徑強制注入成功")
    except Exception as e:
        print(f"⚠️ [環境] 路徑注入警告: {e}")

os.environ["PATH"] = conda_bin_path + os.pathsep + cudnn_bin_path + os.pathsep + os.environ["PATH"]

# ==========================================
# 2. 翻譯參數設定區
# ==========================================
INPUT_SRT = "output.srt"       # 你的中文 SRT 檔案路徑
OUTPUT_SRT = "output_translated.srt"   # 翻譯後的 SRT 輸出路徑

# 可自由修改為："日文 (Japanese)"、"韓文 (Korean)"、"英文 (English)" 
TARGET_LANGUAGE = "英文 (English)"  

# ==========================================
# 3. 讀取並解析原始 SRT 檔案
# ==========================================
def load_srt_blocks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip().replace("\r\n", "\n")
        
    # 依據空行拆分每個字幕區塊
    raw_blocks = content.split("\n\n")
    srt_blocks = []
    
    for block in raw_blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            idx = lines[0].strip()
            time_axis = lines[1].strip()
            text = "\n".join(lines[2:]).strip()
            srt_blocks.append({"idx": idx, "time": time_axis, "text": text})
            
    return srt_blocks

if not os.path.exists(INPUT_SRT):
    print(f"❌ 找不到輸入檔案 {INPUT_SRT}，請確認檔名與路徑。")
    sys.exit()

srt_blocks = load_srt_blocks(INPUT_SRT)
print(f"📂 成功讀取 {INPUT_SRT}，共計 {len(srt_blocks)} 個字幕區塊。")

# ==========================================
# 4. 載入 Qwen2.5-7B-Instruct 翻譯大模型
# ==========================================
print(f"\n🚀 正在載入 Qwen2.5-7B-Instruct 至 3090 Ti (開啟 4-bit 量化)...")
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
# 4-bit 量化僅佔約 6~7GB 顯存，速度極快且能精準保留 CJK 語系的能力
from transformers import BitsAndBytesConfig  # <-- 記得在上方或這裡導入這個類別

# # 正確的 4-bit 量化設定
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True
# )

# print(f"\n🚀 正在透過 BitsAndBytes 載入 Qwen2.5-7B-Instruct 至 3090 Ti...")
# llm_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quantization_config,  # <-- 改用 config 傳入
#     device_map="auto"
# )

# # 切換為高品質的 8-bit 量化設定
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True  # 啟用 8-bit，保留極高精準度
# )

# print(f"\n🚀 正在透過 BitsAndBytes 載入 Qwen2.5-7B-Instruct 至 3090 Ti (開啟高品質 8-bit 量化)...")
# llm_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quantization_config,
#     torch_dtype=torch.float16,  # <-- 關鍵：一定要加這行讓 3090 Ti 起飛
#     device_map="auto"
# )

# 💡 1. 註解或刪除原本的 BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print(f"\n🚀 [核心切換] 3090 Ti 顯存充足 (24GB)，強開原生 bfloat16 完整精度加速推論...")

# 💡 2. 改用純原生 BF16 載入模型（完全繞過 bitsandbytes 的 Windows DLL 陷阱）
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 3090 Ti 原生硬體加速型態
    device_map="auto"            # 自動分派到 GPU
)

# ==========================================
# 5. 批次處理翻譯任務
# ==========================================
print(f"⏳ 開始將字幕精準翻譯為【{TARGET_LANGUAGE}】...")

# 💡 確保 Tokenizer 有設定 padding token，否則無法並行打包
tokenizer.pad_token = tokenizer.eos_token

with open(OUTPUT_SRT, "w", encoding="utf-8") as f_out:
    # 💡 真正的獨立 Batch Size。3090 Ti 跑 8 效能非常卓越
    true_batch_size = 8 
    total_blocks = len(srt_blocks)
    
    system_prompt = f"你是一個資深的資工與電機學術翻譯官。請將用戶提供的 SRT 字幕精準翻譯成【{TARGET_LANGUAGE}】。\n" \
                    f"請絕對遵守以下原則：\n" \
                    f"1. 必須完美保持原有的 SRT 編號、時間軸格式（-->）與換行符號（\\n）完全不變。\n" \
                    f"2. 專業術語如 'Git push', 'GitHub', 'AGILab', 'ASIC', 'FPGA' 必須維持原英文，不可翻譯。"

    print(f"🚀 [多通道並行加速] 3090 Ti 以 {true_batch_size} 獨立 Batch 同步推論中...\n")
    start_time = time.time()  # 紀錄起始時間

    for i in range(0, total_blocks, true_batch_size):
        batch = srt_blocks[i:i+true_batch_size]
        
        batch_prompts = []
        for block in batch:
            single_srt_text = f"{block['idx']}\n{block['time']}\n{block['text']}\n\n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": single_srt_text}
            ]
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(formatted_text)
            
        # 進行批量 Tokenize（此時會自動在左側進行 padding）
        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        # 進行並行生成
        generated_ids = llm_model.generate(
            **model_inputs, 
            max_new_tokens=256,
            do_sample=False, 
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 批量解碼
        input_lens = model_inputs.input_ids.shape[1]
        actual_generated_ids = generated_ids[:, input_lens:]
        responses = tokenizer.batch_decode(actual_generated_ids, skip_special_tokens=True)
        
        # 寫入檔案
        for response in responses:
            f_out.write(response.strip() + "\n\n")
        f_out.flush()
        
        # ==========================================
        # 📊 剩餘時間 (ETA) 與 速度動態計算
        # ==========================================
        current_done = min(i + true_batch_size, total_blocks)
        elapsed_time = time.time() - start_time       # 總花費時間
        blocks_per_sec = current_done / elapsed_time  # 平均每秒處理幾個字幕區塊
        
        remaining_blocks = total_blocks - current_done
        estimated_remaining_time = remaining_blocks / blocks_per_sec  # 預估剩餘秒數
        
        # 將時間格式化為 分:秒
        min_elapsed, sec_elapsed = divmod(int(elapsed_time), 60)
        min_rem, sec_rem = divmod(int(estimated_remaining_time), 60)
        
        # 組合動態進度條（\r 可以讓它在同一行不斷刷新）
        progress_str = f"▓ 進度: {current_done}/{total_blocks} ({current_done/total_blocks*100:.1f}%) " \
                       f"| 已用: {min_elapsed:02d}:{sec_elapsed:02d} " \
                       f"| 剩餘: {min_rem:02d}:{sec_rem:02d} " \
                       f"| 速度: {blocks_per_sec:.2f} 區塊/秒"
        
        print(progress_str, end="\r", flush=True)

print(f"\n\n🎉 字幕翻譯完畢！已將結果輸出至: {OUTPUT_SRT}")