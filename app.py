import os
import time
import json
import sqlite3
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, g
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask application
app = Flask(__name__)

# Configure folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'detection.db')
COCO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8m.pt')
FIREARM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'firearm_detection.pt')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# Database Helper Functions
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        schema_path = os.path.join(BASE_DIR, 'database', 'schema.sql')
        if os.path.exists(schema_path):
            with open(schema_path, mode='r') as f:
                db.cursor().executescript(f.read())
            db.commit()
            print("Database initialized successfully.")
        else:
            print("Database schema.sql not found!")

# Initialize database
init_db()

# Load YOLOv8 Models (downloads model weights to models/ folder if not present)
print(f"Loading COCO model from {COCO_MODEL_PATH}...")
model_coco = YOLO(COCO_MODEL_PATH)
print(f"Loading Firearm model from {FIREARM_MODEL_PATH}...")
model_firearm = YOLO(FIREARM_MODEL_PATH)
print("Models loaded successfully.")

# Helper function to generate class-specific neon colors
def get_class_color(class_id):
    np.random.seed(class_id + 5)
    color = np.random.randint(50, 255, size=3).tolist()
    return tuple(color)

# Page Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

# REST API Endpoints
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        file = request.files.get('image')
        webcam_json = request.get_json(silent=True)
        webcam_data = webcam_json.get('webcam_image') if webcam_json else None

        img_bytes = None
        original_filename = ""

        if file:
            original_filename = file.filename
            img_bytes = file.read()
        elif webcam_data:
            original_filename = f"webcam_{uuid.uuid4().hex[:8]}.jpg"
            if ',' in webcam_data:
                webcam_data = webcam_data.split(',')[1]
            img_bytes = base64.b64decode(webcam_data)
        else:
            return jsonify({'error': 'No image data provided'}), 400

        unique_id = uuid.uuid4().hex
        filename = f"{unique_id}_orig.jpg"
        output_filename = f"{unique_id}_pred.jpg"
        
        orig_path = os.path.join(UPLOAD_FOLDER, filename)
        pred_path = os.path.join(OUTPUT_FOLDER, output_filename)

        with open(orig_path, 'wb') as f:
            f.write(img_bytes)

        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as img_err:
            print("Failed to read image via Pillow:", img_err)
            img = cv2.imread(orig_path)
            if img is None:
                return jsonify({'error': 'Invalid image format', 'message': str(img_err)}), 400

        # Run YOLOv8 inferences
        results_coco = model_coco(img)
        result_coco = results_coco[0]
        
        results_firearm = model_firearm(img)
        result_firearm = results_firearm[0]
        
        inference_time_ms = float(result_coco.speed.get('inference', 0.0)) + float(result_firearm.speed.get('inference', 0.0))
        if inference_time_ms == 0.0:
            inference_time_ms = (time.time() - start_time) * 1000

        detections = []
        confidences = []
        annotated_img = img.copy()
        
        # 1. Parse Firearm Detections First
        gun_boxes = []
        for box in result_firearm.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = "gun"
            
            # Enforce 70% threshold on firearm model to suppress false positive background objects
            if conf < 0.70:
                continue
                
            gun_boxes.append((x1, y1, x2, y2))
            confidences.append(conf)
            color = (0, 0, 255) # Red for danger/weapons
            
            label_text = f"{label.capitalize()} {int(conf * 100)}%"
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_img, 
                (x1, y1 - text_height - 10 if y1 - text_height - 10 > 0 else 0),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            cv2.putText(
                annotated_img,
                label_text,
                (x1 + 5, y1 - 5 if y1 - text_height - 10 > 0 else text_height + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            detections.append({
                'name': label,
                'confidence': round(conf, 4),
                'box': [x1, y1, x2, y2]
            })

        # 2. Parse COCO Detections
        raw_coco_detections = []
        for box in result_coco.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model_coco.names[class_id]
            
            # Heuristic: If vase is tall and narrow, it is a thermos/water bottle
            if label == "vase":
                h = y2 - y1
                w = x2 - x1
                if w > 0 and (h / w) > 2.0:
                    label = "bottle"
            # Heuristic: If TV is detected but its dominant color inside the box is green, it is a chalkboard/blackboard
            elif label == "tv":
                if y2 > y1 and x2 > x1:
                    avg_color = cv2.mean(img[y1:y2, x1:x2])[:3]
                    # If Green component is significantly higher than Red and Blue, it's a chalkboard
                    if avg_color[1] > avg_color[2] * 1.5 and avg_color[1] > avg_color[0] * 1.3:
                        label = "blackboard"
            # Heuristic: If Oven is detected with low confidence and is dark, it is a chalkboard easel
            elif label == "oven":
                if y2 > y1 and x2 > x1 and conf < 0.65:
                    avg_color = cv2.mean(img[y1:y2, x1:x2])[:3]
                    if avg_color[0] < 100 and avg_color[1] < 100 and avg_color[2] < 100:
                        label = "blackboard easel"
            
            raw_coco_detections.append({
                'name': label,
                'confidence': conf,
                'class_id': class_id,
                'box': [x1, y1, x2, y2]
            })

        # Filter out overlapping/spurious detections
        has_blackboard = any("blackboard" in det['name'] for det in raw_coco_detections)
        blackboard_boxes = [det['box'] for det in raw_coco_detections if "blackboard" in det['name']]

        for det in raw_coco_detections:
            label = det['name']
            conf = det['confidence']
            class_id = det['class_id']
            x1, y1, x2, y2 = det['box']

            # Suppress bench detections that are easel legs underneath a blackboard
            if label == "bench" and has_blackboard:
                is_easel_leg = False
                for bx1, by1, bx2, by2 in blackboard_boxes:
                    # Check X-axis overlap
                    overlap_x = max(0, min(x2, bx2) - max(x1, bx1))
                    bench_w = x2 - x1
                    # If bench X-range overlaps significantly with blackboard X-range, and is near the bottom
                    if bench_w > 0 and (overlap_x / bench_w) > 0.5:
                        if y1 < by2 + 80 and y2 > by1:
                            is_easel_leg = True
                            break
                if is_easel_leg:
                    continue

            # Filter out overlapping false-positives (except person) if a gun is detected
            if label != "person" and gun_boxes:
                overlap = False
                for gx1, gy1, gx2, gy2 in gun_boxes:
                    ix1 = max(x1, gx1)
                    iy1 = max(y1, gy1)
                    ix2 = min(x2, gx2)
                    iy2 = min(y2, gy2)
                    if ix1 < ix2 and iy1 < iy2:
                        int_area = (ix2 - ix1) * (iy2 - iy1)
                        box_area = (x2 - x1) * (y2 - y1)
                        if int_area / box_area > 0.4: # Overlaps > 40%
                            overlap = True
                            break
                if overlap:
                    continue

            confidences.append(conf)
            color = get_class_color(class_id)
            label_text = f"{label.capitalize()} {int(conf * 100)}%"
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_img, 
                (x1, y1 - text_height - 10 if y1 - text_height - 10 > 0 else 0),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            cv2.putText(
                annotated_img,
                label_text,
                (x1 + 5, y1 - 5 if y1 - text_height - 10 > 0 else text_height + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            detections.append({
                'name': label,
                'confidence': round(conf, 4),
                'box': [x1, y1, x2, y2]
            })

        cv2.imwrite(pred_path, annotated_img)

        objects_count = len(detections)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO history (filename, output_filename, objects_count, objects_list, average_confidence, inference_time, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                output_filename,
                objects_count,
                json.dumps(detections),
                round(avg_confidence, 4),
                round(inference_time_ms, 2),
                "YOLOv8m + Custom"
            )
        )
        db.commit()
        new_id = cursor.lastrowid

        response_payload = {
            'id': new_id,
            'filename': f'/static/uploads/{filename}',
            'output_filename': f'/static/outputs/{output_filename}',
            'objects_count': objects_count,
            'objects_list': detections,
            'average_confidence': round(avg_confidence * 100, 2),
            'inference_time': round(inference_time_ms, 2),
            'model_name': 'YOLOv8m + Custom',
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(response_payload)

    except Exception as e:
        print("Prediction API Error:", e)
        return jsonify({'error': 'Server inference failed', 'message': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        search_query = request.args.get('q', '').strip().lower()
        limit = int(request.args.get('limit', 50))
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        history_list = []
        for r in rows:
            record = dict(r)
            record['filename'] = f'/static/uploads/{record["filename"]}'
            record['output_filename'] = f'/static/outputs/{record["output_filename"]}'
            record['objects_list'] = json.loads(record['objects_list'])
            
            if search_query:
                object_names = [obj['name'].lower() for obj in record['objects_list']]
                match_found = any(search_query in name for name in object_names)
                if not match_found:
                    continue
                    
            history_list.append(record)
            
        return jsonify(history_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:id>', methods=['DELETE'])
def delete_history_item(id):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT filename, output_filename FROM history WHERE id = ?", (id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'error': 'Record not found'}), 404
            
        record = dict(row)
        orig_file = os.path.join(UPLOAD_FOLDER, record['filename'])
        pred_file = os.path.join(OUTPUT_FOLDER, record['output_filename'])
        
        cursor.execute("DELETE FROM history WHERE id = ?", (id,))
        db.commit()
        
        if os.path.exists(orig_file):
            os.remove(orig_file)
        if os.path.exists(pred_file):
            os.remove(pred_file)
            
        return jsonify({'ok': True, 'message': 'Record deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) as total_images, SUM(objects_count) as total_objects, AVG(average_confidence) as avg_conf FROM history")
        kpis = dict(cursor.fetchone())
        
        total_images = kpis['total_images'] or 0
        total_objects = kpis['total_objects'] or 0
        avg_confidence = round((kpis['avg_conf'] or 0.0) * 100, 2)
        
        cursor.execute("SELECT objects_list FROM history")
        all_notes = cursor.fetchall()
        
        class_distribution = {}
        for row in all_notes:
            objects = json.loads(row[0])
            for obj in objects:
                name = obj['name'].capitalize()
                class_distribution[name] = class_distribution.get(name, 0) + 1
                
        most_detected = "-"
        if class_distribution:
            most_detected = max(class_distribution, key=class_distribution.get)
            
        cursor.execute(
            """
            SELECT date(created_at) as date, COUNT(*) as count, SUM(objects_count) as obj_count 
            FROM history 
            GROUP BY date(created_at) 
            ORDER BY date(created_at) ASC 
            LIMIT 30
            """
        )
        timeline_rows = cursor.fetchall()
        timeline_data = [dict(r) for r in timeline_rows]
        
        sorted_distribution = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
        distribution_labels = [item[0] for item in sorted_distribution]
        distribution_values = [item[1] for item in sorted_distribution]
        
        stats_payload = {
            'total_images': total_images,
            'total_objects': total_objects,
            'most_detected': most_detected,
            'average_confidence': avg_confidence,
            'distribution': {
                'labels': distribution_labels,
                'values': distribution_values
            },
            'timeline': timeline_data
        }
        
        return jsonify(stats_payload)
    except Exception as e:
        print("Stats API error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
