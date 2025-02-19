from flask import Flask, request, render_template, send_from_directory, jsonify
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


def predict_yolov8(input_path, file):
    try:
        model = YOLO(r'D:\Program Files\road_snow\yolov8n\runs\detect\train310\weights\best.pt')
        results = model.predict(
            source=input_path,
            project=app.config['RESULTS_FOLDER'],
            name=os.path.splitext(file.filename)[0],
            save=True,
            exist_ok=True
        )

        result_dir = str(results[0].save_dir)
        result_files = [f for f in os.listdir(result_dir) if f.endswith(('.avi', '.jpg', '.png'))]

        if not result_files:
            raise RuntimeError("No output file generated.")

        result_filename = result_files[0]

        # 确保 ffmpeg 在系统 PATH 中
        if result_filename.endswith('.avi'):
            result_filename_mp4 = os.path.splitext(result_filename)[0] + '.mp4'
            avi_path = os.path.join(result_dir, result_filename)
            mp4_path = os.path.join(result_dir, result_filename_mp4)

            # 调用 ffmpeg 进行转换
            subprocess.run(['ffmpeg', '-i', avi_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', mp4_path], check=True)

            os.remove(avi_path)  # 删除原 .avi 文件
            result_filename = result_filename_mp4

        print(f"Result saved to: {result_dir}\\{result_filename}")
        return f'/static/results/{os.path.basename(result_dir)}/{result_filename}'

    except Exception as e:
        print(f"Error in predict_yolov8: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 安全处理文件名
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(input_path)
        result_url = predict_yolov8(input_path, file)

        if not result_url:
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({'result': result_url})

    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
