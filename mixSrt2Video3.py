import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import subprocess
import gc
import time

# 1. 設定路徑與切換工作目錄
current_dir = r"C:\github\VideoProcess\faster-whisper"
os.chdir(current_dir)

video_input = "2026_Github01_raw1.mp4"
srt_input = "output.srt"
video_output = "output_fast.mp4"

def time2sec(t):
    arr = t.split(' --> ')
    s1 = arr[0].split(',')
    s2 = arr[1].split(',')
    start = int(s1[0].split(':')[0])*3600 + int(s1[0].split(':')[1])*60 + int(s1[0].split(':')[2]) + float(s1[1])*0.001
    end = int(s2[0].split(':')[0])*3600 + int(s2[0].split(':')[1])*60 + int(s2[0].split(':')[2]) + float(s2[1])*0.001
    return [start, end]

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fmt_eta(seconds):
    if seconds < 60: return f"{int(seconds)}秒"
    elif seconds < 3600: return f"{int(seconds//60)}分{int(seconds%60)}秒"
    else: return f"{int(seconds // 3600)}小時{int((seconds % 3600) // 60)}分"

# 2. 解析 SRT 字幕
print("正在解析字幕檔...")
with open(srt_input, 'r', encoding='utf-8') as f:
    srt_list = f.read().split('\n')

subtitles = []
sec, text = 1, 2
for i in range(len(srt_list)):
    if i == sec:
        sec += 4
        subtitles.append({"range": time2sec(srt_list[i]), "text": ""})
    if i == text:
        text += 4
        if subtitles:
            subtitles[-1]["text"] = srt_list[i]

# 3. 讀取影片資訊
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 4. 字體設定
base_path = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(base_path, 'assets', 'fonts', 'kaiu.ttf')
font_size = int(height * 0.045)
font = ImageFont.truetype(font_path, font_size)
max_text_width = int(width * 0.8)

# 5. 💡 修正編碼器選擇：既然此環境 FFmpeg 不支援 NVENC，我們調用多核心相容性最高的 libx264
# 並開啟 -preset ultrafast，把 CPU 解壓與壓縮開銷降到極低，強迫硬推速度
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',              # 使用多核優化的 x264
    '-preset', 'ultrafast',         # 極速模式，完全不拖累 Python 快取流
    '-tune', 'zerolatency',         # 零延遲吞吐
    '-b:v', '5M',
    '-pix_fmt', 'yuv420p',
    video_output
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
print(f"影片解析度: {width}x{height}, 總幀數: {total_frames}")
print("高效 NumPy 矩陣快取版啟動，開始全速渲染...\n")

def get_wrapped_lines(text, draw_obj):
    lines = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        bbox = draw_obj.textbbox((0, 0), test_line, font=font)
        if (bbox[2] - bbox[0]) <= max_text_width:
            current_line = test_line
        else:
            if current_line: lines.append(current_line)
            current_line = char
    if current_line: lines.append(current_line)
    return lines

# 💡 精簡版透明字卡產生器（只產出緊湊的字幕區域，不浪費記憶體畫整張空圖）
def create_compact_subtitle(text):
    # 先隨便建立一個小畫布用來測量行數
    temp_img = Image.new('RGBA', (100, 100))
    temp_draw = ImageDraw.Draw(temp_img)
    lines = get_wrapped_lines(text, temp_draw)
    
    line_height = font_size + int(font_size * 0.2)
    sub_h = line_height * len(lines) + 20
    sub_w = width
    
    # 建立一個剛好涵蓋字幕區域的畫布
    overlay = Image.new('RGBA', (sub_w, sub_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    start_y = 10
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (sub_w - line_w) // 2
        y = start_y + (idx * line_height)
        draw.text((x, y), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill='black')
        
    # 轉成 OpenCV 的 BGR 與 Alpha 遮罩
    arr = np.array(overlay)
    sub_bgr = arr[:, :, :3]
    sub_bgr = cv2.cvtColor(sub_bgr, cv2.COLOR_RGB2BGR)
    alpha = arr[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2) # 變成 (h, w, 1) 方便做矩陣相乘
    
    # 計算這個字卡要貼在影片的哪個 Y 軸區間
    target_y1 = int(height * 0.85) - sub_h
    target_y2 = int(height * 0.85)
    
    return sub_bgr, alpha, target_y1, target_y2

# 字幕快取字典
subtitle_cache = {}
frame_count = 0
start_run_time = time.time()

# 6. 核心極速矩陣切片迴圈
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    current_time = frame_count / fps
    current_text = ""
    for sub in subtitles:
        if sub["range"][0] <= current_time <= sub["range"][1]:
            current_text = sub["text"]
            break
            
    if current_text:
        if current_text not in subtitle_cache:
            subtitle_cache[current_text] = create_compact_subtitle(current_text)
            
        # 💡 核心優化：直接抓出快取的 BGR 像素與 Alpha 遮罩
        sub_bgr, alpha, y1, y2 = subtitle_cache[current_text]
        
        # 僅針對影片底部的字幕區域做 NumPy 矩陣融合，其餘大面積畫面完全不運算！
        roi = frame[y1:y2, 0:width]
        blended = (sub_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
        frame[y1:y2, 0:width] = blended
    
    # 拋入管道
    proc.stdin.write(frame.tobytes())
    frame_count += 1
    
    if frame_count % 10000 == 0:
        gc.collect()
        
    if frame_count % 500 == 0:
        elapsed_run_time = time.time() - start_run_time
        current_fps = frame_count / elapsed_run_time
        remaining_frames = total_frames - frame_count
        eta_seconds = remaining_frames / current_fps if current_fps > 0 else 0
        
        print(f"進度: {frame_count}/{total_frames} 影格 ({(frame_count/total_frames)*100:.2f}%) | 影片當前秒數: {fmt_time(current_time)} | 速度: {current_fps:.1f} fps | 預計還需: {fmt_eta(eta_seconds)}      ", end="\r")

cap.release()
proc.stdin.close()
proc.wait()
cv2.destroyAllWindows()
subtitle_cache.clear()
gc.collect()

# 7. 後處理：無損音軌貼回
print("\n\n影像渲染完成！正在進行最後的音軌無損合併...")
try:
    from moviepy import VideoFileClip
    orig_video = VideoFileClip(video_input)
    rendered_video = VideoFileClip(video_output)
    if orig_video.audio is not None:
        final_video = rendered_video.with_audio(orig_video.audio)
        final_video.write_videofile("final_output.mp4", codec="libx264", audio_codec="copy", logger=None)
        print("🎉 大功告成！請檢查最終檔案: final_output.mp4")
        rendered_video.close()
        orig_video.close()
        os.remove(video_output)
except Exception as e:
    print(f"音訊合併失敗，無聲影片已保存在 {video_output}。原因: {e}")