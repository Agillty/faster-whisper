
#%%
import os
from moviepy import VideoFileClip
from moviepy import VideoFileClip, vfx
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy import concatenate_videoclips

from PIL import Image, ImageFont, ImageDraw

# 定義轉換為總秒數的函式
def time2sec(t):
    arr = t.split(' --> ')   # 根據「' --> '」拆分文字
    s1 = arr[0].split(',')   # 前方的文字為開始時間
    s2 = arr[1].split(',')   # 後方的文字為結束時間
    # 計算開始時間的總秒數
    start = int(s1[0].split(':')[0])*3600 + int(s1[0].split(':')[1])*60 + int(s1[0].split(':')[2]) + float(s1[1])*0.001
    # 計算結束時間的總秒數
    end = int(s2[0].split(':')[0])*3600 + int(s2[0].split(':')[1])*60 + int(s2[0].split(':')[2]) + float(s2[1])*0.001
    return [start, end]      # 回傳開始時間與結束時間的串列

f = open('output.srt','r')  # 使用 open 方法的 r 開啟字幕檔案
srt = f.read()                  # 讀取字幕檔案內容
f.close()                       # 關閉檔案
srt_list = srt.split('\n')      # 將內容根據換行符號 \n 拆分成串列
sec = 1                         # 串列中秒數從第二項開始 ( 串列的第二項的索引值為 1 )
text = 2                        # 串列中文字內容從第三項開始 ( 串列的第三項的索引值為 2 )
sec_list = [[0,0]]              # 定義時間串列的開頭為 [0,0]
text_list = ['']                # 定義字幕內容串列的開頭為空字串 ''
# 使用迴圈，讀取字幕檔案串列的每個項目
for i in range(len(srt_list)):
    if i == sec:
        sec = sec + 4           # 如果遇到時間內容，就將 sec + 4 ( 因為時間每隔 4 個項目會出現 )
        # 如果兩個串列項目內容前後對不上 ( 前一個結束時間不等於後一個的開始時間 )
        if sec_list[-1][1] != time2sec(srt_list[i])[0]:
            # 在時間串列中間添加一個開始時間與結束時間內容 ( 表示該區間沒有字幕 )
            sec_list.append([sec_list[-1][1],time2sec(srt_list[i])[0]])
            # 在文字串列中間添加一個空字串 ( 表示該區間沒有字幕 )
            text_list.append('')
        sec_list.append(time2sec(srt_list[i]))  # 添加時間到時間串列
    if i == text:
        text = text + 4               # 如果遇到文字內容，就將 text + 4 ( 因為文字每隔 4 個項目會出現 )
        text_list.append(srt_list[i]) # 添加文字到文字串列

print(sec_list)
print(text_list)
# %%
# font = ImageFont.truetype('NotoSansTC-Regular.otf', 20)   # 設定文字字體和大小

base_path = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(base_path, 'assets', 'fonts', 'kaiu.ttf')
font = ImageFont.truetype(font_path, 20)

# video = VideoFileClip("2026_Github01_raw1.mp4").with_effects([vfx.Resize((480, 240))])
video = VideoFileClip("2026_Github01_raw1.mp4") # 讀取影片
w, h = video.size  # 動態獲取影片寬高

video_duration = float(video.duration)                    # 讀取影片總長度
output_list = []                                          # 記錄最後要組合的影片片段

# 如果字幕最後的時間小於總長度
if sec_list[-1][1] != video_duration:
    sec_list.append([sec_list[-1][1],video_duration])     # 添加時間到時間串列
    text_list.append('')                                  # 添加空字串到文字串列

# 建立文字字卡函式
def text_clip(text, name,video_w, video_h):
# 產生與影片同大小的透明背景
    img = Image.new('RGBA', (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 動態計算文字大小與位置 (設定字體大小為高度的 1/20)
    font_size = int(video_h * 0.05)
    font = ImageFont.truetype(font_path, font_size)
    
    # 計算文字寬度 (估算)
    text_width = font_size * len(text)
    
    # 將文字放在底部 80% 的位置
    draw.text(((video_w - text_width) / 2, video_h * 0.8), 
              text, fill=(255, 255, 255), font=font, 
              stroke_width=2, stroke_fill='black')
    img.save(name)

# 建立影片和文字合併的函式
def text_in_video(t, text_img):
    # clip = video.subclip(t[0],t[1])                  # 剪輯影片到指定長度
    clip = video.subclipped(t[0], t[1])
    # text = ImageClip(text_img, transparent=True).set_duration(t[1]-t[0])  # 讀取字卡，調整為影片長度
    text = ImageClip(text_img).with_duration(t[1] - t[0])
    combine_clip = CompositeVideoClip([clip, text])  # 合併影片和文字
    output_list.append(combine_clip)                 # 添加到影片片段裡

# 使用 for 迴圈，產生文字字卡
for i in range(len(text_list)):
    text_clip(text_list[i], 'srt.png',w,h)
    text_in_video(sec_list[i], 'srt.png')

output = concatenate_videoclips(output_list)      # 合併所有影片片段
output.write_videofile("output.mp4",temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
print('ok')
# %%
