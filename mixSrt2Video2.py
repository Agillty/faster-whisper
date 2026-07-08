import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import subprocess
import gc
import time  # 💡 新增：用來精確計算時間與 ETA

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
    """將秒數格式化為 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fmt_eta(seconds):
    """將剩餘秒數格式化為更直觀的 X小時X分X秒"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds//60)}分{int(seconds%60)}秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}小時{m}分{s}秒"

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

# 5. 配置硬體加速編碼器管道
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'h264_nvenc',
    '-preset', 'p4',
    '-b:v', '6M',
    '-pix_fmt', 'yuv420p',
    video_output
]

proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

print(f"影片解析度: {width}x{height}, 總幀數: {total_frames}")
print("3090 Ti Pipe 硬體加速快取版啟動，開始全速渲染...\n")

# 自動換行輔助
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

# 建立單句字幕的透明字卡底圖
def create_subtitle_overlay(text):
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    lines = get_wrapped_lines(text, draw)
    line_height = font_size + int(font_size * 0.2)
    start_y = int(height * 0.85) - (len(lines) * line_height)
    
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (width - line_w) // 2
        y = start_y + (idx * line_height)
        draw.text((x, y), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill='black')
    
    return overlay

# 初始化計時器與統計變數
subtitle_cache = {}
frame_count = 0
start_run_time = time.time()  # 紀錄大迴圈開始執行的絕對時間

# 6. 核心 Pipe 寫入迴圈
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
            subtitle_cache[current_text] = create_subtitle_overlay(current_text)
            
        sub_overlay = subtitle_cache[current_text]
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
        combined = Image.alpha_composite(img_pil, sub_overlay)
        frame = cv2.cvtColor(np.array(combined.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # 將影格寫入 FFmpeg 管道
    proc.stdin.write(frame.tobytes())
    frame_count += 1
    
    if frame_count % 10000 == 0:
        gc.collect()
        
    # 💡 這裡每 500 幀更新一次動態進度（提升即時感，且不影響效能）
    if frame_count % 500 == 0:
        elapsed_run_time = time.time() - start_run_time
        current_fps = frame_count / elapsed_run_time
        
        # 計算剩餘影格與預期完成時間 (ETA)
        remaining_frames = total_frames - frame_count
        eta_seconds = remaining_frames / current_fps if current_fps > 0 else 0
        
        pct = (frame_count / total_frames) * 100
        vid_time_str = fmt_time(current_time)
        eta_str = fmt_eta(eta_seconds)
        
        # 輸出漂亮的單行進度條
        print(f"進度: {frame_count}/{total_frames} 影格 ({pct:.2f}%) | 影片當前秒數: {vid_time_str} | 速度: {current_fps:.1f} fps | 預計還需: {eta_str}      ", end="\r")

# 關閉資源
cap.release()
proc.stdin.close()
proc.wait()
cv2.destroyAllWindows()

subtitle_cache.clear()
gc.collect()

# 7. 後處理：複製音軌
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