import os
import subprocess

# 1. 設定工作目錄
current_dir = r"C:\github\VideoProcess\faster-whisper"
os.chdir(current_dir)

# 🔍 鐵證檢查：印出目前目錄下的所有檔案，看 ffmpeg 到底躲在哪裡
print("=== 📂 目前資料夾內的檔案列表 ===")
files = os.listdir(current_dir)
for f in files:
    print(f" - {f}")
print("=================================\n")

ffmpeg_exe_path = os.path.join(current_dir, "ffmpeg.exe")

# 檢查檔案是否真的存在
if not os.path.exists(ffmpeg_exe_path):
    print(f"❌ 警告：在路徑 {ffmpeg_exe_path} 找不到 ffmpeg.exe！")
    print("💡 請檢查你下載的完全體 FFmpeg 是不是解壓到子資料夾（例如 bin 裡面）了？")
    print("💡 請把 ffmpeg.exe 複製到 C:\\github\\VideoProcess\\faster-whisper 再執行一次！")
    exit(1)

video_input = "2026_Github01_raw1.mp4"
srt_input = "output.srt"
video_output = "final_output_gpu.mp4"

# 💡 Windows 字幕路徑修正
srt_filter_path = os.path.abspath(srt_input).replace('\\', '/').replace(':', '\\:')

# 2. 指令列表
ffmpeg_cmd = [
    ffmpeg_exe_path,
    "-y",
    "-hwaccel", "cuda", 
    "-i", video_input,
    "-vf", f"subtitles='{srt_filter_path}'",
    "-c:v", "h264_nvenc",
    "-preset", "p4",
    "-b:v", "5M",
    "-c:a", "copy",
    video_output
]

print("🚀 3090 Ti 列表指令版點火...")
print(f"👉 執行指令: {' '.join(ffmpeg_cmd)}\n")

# 3. 啟動程序
process = subprocess.Popen(
    ffmpeg_cmd, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    universal_newlines=True, 
    encoding='cp950',
    errors='ignore',
    shell=False
)

# 4. 監控輸出
try:
    for line in process.stdout:
        line_str = line.strip()
        if not line_str: continue
        
        if "frame=" in line_str or "time=" in line_str or "speed=" in line_str:
            print(line_str, end="\r")
        else:
            print(f"[FFmpeg 輸出]: {line_str}")
                
except Exception as e:
    print(f"\n讀取輸出時發生錯誤: {e}")

process.wait()

if process.returncode == 0:
    print("\n\n🎉 渲染成功！")
else:
    print(f"\n\n❌ 執行失敗，錯誤碼: {process.returncode}")