# AI Object Detection Web Application

A modern full-stack web application that allows users to upload images or stream live webcam video to detect and classify objects in real-time. Built with a Flask backend, OpenCV for image processing, SQLite for history logging, and a glassmorphic dashboard frontend.

## Tech Stack

* **Backend:** Python, Flask, OpenCV, SQLite, NumPy, Pillow, Ultralytics YOLOv8
* **Frontend:** HTML5, CSS3, JavaScript (ES6), Bootstrap 5, Chart.js, Font Awesome

## Getting Started (F Drive Setup)

To run this application without running out of disk space on your C drive, follow these instructions:

### 1. Set up a Python Virtual Environment on the F Drive
Creating a virtual environment inside the `F:\ObjectDetectionAI` folder ensures that large dependencies (like PyTorch and OpenCV, which occupy over 2.5 GB of space) are stored directly on your F drive instead of C.

Open your terminal and run:
```powershell
cd "F:\ObjectDetectionAI"
python -m venv venv
```

### 2. Activate the Virtual Environment
```powershell
.\venv\Scripts\activate
```

### 3. Install Dependencies
Make sure you point pip's cache and build folder to the F drive to prevent any out-of-disk-space errors during compilation:
```powershell
$env:TMP="F:\temp"
$env:TEMP="F:\temp"
pip install --cache-dir="F:\pip-cache" -r requirements.txt
```

### 4. Run the Flask App
```powershell
python app.py
```

Access the application in your browser at `http://localhost:5000`.
