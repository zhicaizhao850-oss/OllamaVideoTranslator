import os
import json
import uuid
import subprocess
import threading
import webbrowser
import shutil
from flask import Flask, request, jsonify, render_template, send_file
# AI Agent: 导入 faster_whisper（已升级到 1.2.1 版本，支持 av>=11）
from faster_whisper import WhisperModel
# AI Agent: 导入繁简转换库
try:
    import zhconv
    ZHCONV_AVAILABLE = True
except ImportError:
    ZHCONV_AVAILABLE = False
    print("警告: zhconv 未安装，无法进行繁简转换。请运行: pip install zhconv")
import ollama_processor

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'models/faster-whisper-base'

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 全局变量存储模型
whisper_model = None

# AI Agent: 查找 FFmpeg 可执行文件路径
def find_ffmpeg():
    """查找 FFmpeg 可执行文件路径"""
    # 首先尝试使用 shutil.which 查找（会在 PATH 中查找）
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # 如果找不到，尝试常见路径
    common_paths = [
        r'F:\Ffmpeg\ffmpeg\bin\ffmpeg.exe',  # 根据 where.exe 的结果
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None

# 初始化时查找 FFmpeg 路径
FFMPEG_PATH = find_ffmpeg()
if not FFMPEG_PATH:
    print("警告: 未找到 FFmpeg，请确保 FFmpeg 已安装并添加到 PATH 环境变量")
else:
    print(f"找到 FFmpeg: {FFMPEG_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_model')
def check_model():
    """检查模型是否已下载"""
    model_path = MODEL_PATH
    if os.path.exists(model_path):
        return jsonify({'downloaded': True})
    return jsonify({'downloaded': False})

@app.route('/download_model')
def download_model():
    """下载模型"""
    try:
        # 使用huggingface_hub下载模型
        from huggingface_hub import snapshot_download
        
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            snapshot_download(
                repo_id="Systran/faster-whisper-base",
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'file_id': file_id,
        'filename': filename,
        'filepath': filepath
    })

@app.route('/extract_subtitles', methods=['POST'])
def extract_subtitles():
    """提取字幕"""
    try:
        data = request.json
        file_path = data.get('file_path')
        source_language = data.get('source_language', 'auto')
        output_format = data.get('output_format', 'srt')
        enable_vad = data.get('enable_vad', True)
        enable_gpu = data.get('enable_gpu', False)
        enable_word_timestamps = data.get('enable_word_timestamps', False)
        # AI Agent: 添加繁简转换选项，默认为 True（自动转换为简体中文）
        convert_to_simplified = data.get('convert_to_simplified', True)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'})
        
        # 检查模型
        global whisper_model
        if whisper_model is None:
            model_path = MODEL_PATH
            if not os.path.exists(model_path):
                return jsonify({'error': '模型未下载，请先下载模型'})
            
            device = "cuda" if enable_gpu else "cpu"
            compute_type = "float16" if enable_gpu else "int8"
            
            whisper_model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type
            )
        
        # 生成输出文件名
        file_id = str(uuid.uuid4())
        output_filename = f"{file_id}_subtitles.{output_format}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # 提取音频
        audio_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_audio.wav")
        
        # AI Agent: 使用找到的 FFmpeg 路径，如果找不到则返回错误
        if not FFMPEG_PATH:
            return jsonify({'error': 'FFmpeg 未找到，请确保 FFmpeg 已安装并添加到 PATH 环境变量'})
        
        try:
            result = subprocess.run([
                FFMPEG_PATH, '-i', file_path, '-ar', '16000', '-ac', '1', '-y', audio_path
            ], check=True, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'FFmpeg 处理超时，请检查视频文件是否过大'})
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            return jsonify({'error': f'FFmpeg 处理失败: {error_msg}'})
        except FileNotFoundError:
            return jsonify({'error': f'FFmpeg 未找到: {FFMPEG_PATH}，请检查 FFmpeg 安装路径'})
        
        # 转录
        segments, info = whisper_model.transcribe(
            audio_path,
            language=None if source_language == 'auto' else source_language,
            vad_filter=enable_vad,
            word_timestamps=enable_word_timestamps
        )
        
        # AI Agent: 繁简转换函数
        def convert_text(text):
            """将文本转换为简体中文（如果需要）"""
            if convert_to_simplified and ZHCONV_AVAILABLE:
                # 检测是否为繁体中文，如果是则转换为简体
                return zhconv.convert(text, 'zh-cn')
            return text
        
        # 保存字幕
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_format == 'srt':
                for i, segment in enumerate(segments, 1):
                    start = segment.start
                    end = segment.end
                    text = convert_text(segment.text.strip())
                    
                    f.write(f"{i}\n")
                    f.write(f"{format_time_srt(start)} --> {format_time_srt(end)}\n")
                    f.write(f"{text}\n\n")
            else:
                # 其他格式处理
                f.write("# 字幕内容\n")
                for segment in segments:
                    text = convert_text(segment.text.strip())
                    f.write(f"[{format_time_txt(segment.start)}] {text}\n")
        
        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return jsonify({
            'success': True,
            'output_path': output_path,
            'filename': output_filename,
            'language': info.language,
            'duration': info.duration
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def format_time_srt(seconds):
    """格式化时间为SRT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def format_time_txt(seconds):
    """格式化时间为文本格式"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

@app.route('/check_ollama')
def check_ollama():
    """检查OLLAMA服务状态"""
    return ollama_processor.check_ollama()

@app.route('/get_models')
def get_models():
    """获取OLLAMA模型列表"""
    return ollama_processor.get_models()

@app.route('/optimize_subtitles', methods=['POST'])
def optimize_subtitles():
    """优化字幕"""
    try:
        data = request.json
        subtitle_path = data.get('subtitle_path')
        model = data.get('model', 'llama3.2')
        enable_segment = data.get('enable_segment', True)
        enable_translate = data.get('enable_translate', False)
        target_language = data.get('target_language', 'zh')
        segment_parts = data.get('segment_parts', 1)
        max_sentence_length = data.get('max_sentence_length', 30)
        segment_prompt = data.get('segment_prompt', '')
        translate_format = data.get('translate_format', 'srt')
        output_format = data.get('output_format', 'srt')
        
        if not subtitle_path or not os.path.exists(subtitle_path):
            return jsonify({'error': '字幕文件不存在'})
        
        # 读取字幕内容
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            subtitle_content = f.read()
        
        # 调用OLLAMA处理
        result = ollama_processor.optimize_subtitles(
            subtitle_content,
            model=model,
            enable_segment=enable_segment,
            enable_translate=enable_translate,
            target_language=target_language,
            segment_parts=segment_parts,
            max_sentence_length=max_sentence_length,
            segment_prompt=segment_prompt,
            translate_format=translate_format
        )
        
        if result['success']:
            # 保存优化后的字幕
            file_id = str(uuid.uuid4())
            output_filename = f"{file_id}_optimized.{output_format}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['content'])
            
            return jsonify({
                'success': True,
                'output_path': output_path,
                'filename': output_filename
            })
        else:
            return jsonify({'error': result.get('error', '优化失败')})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    """下载文件"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'})
    except Exception as e:
        return jsonify({'error': str(e)})

def open_browser():
    """打开浏览器"""
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # 延迟1秒后打开浏览器
    threading.Timer(1, open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=5000)