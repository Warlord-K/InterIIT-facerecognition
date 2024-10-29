# Real-time Face Recognition System | [DEMO](https://huggingface.co/spaces/Warlord-K/FaceRecognition-InterIIT)

A Python-based real-time face recognition system that combines YOLO face detection with ArcFace embeddings for accurate face recognition. The system processes webcam feed in real-time, detecting faces and matching them against a stored database of known faces.

## Features

- Real-time face detection using YOLO
- Face recognition using ArcFace embeddings
- Persistent storage of face embeddings
- Interactive interface for adding new faces to the database
- Configurable confidence thresholds for detection and recognition
- Live display of recognition results with bounding boxes and names
- Batch processing tool for building face database from image folders

## Prerequisites

- Python 3.8 or higher
- Webcam or camera device

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Warlord-K/InterIIT-facerecognition.git
cd InterIIT-facerecognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the Arcface model: Click this [link](https://www.digidow.eu/f/datasets/arcface-tensorflowlite/model.tflite) and put it in the same folder as the script

## Usage

### Real-time Recognition

1. Run the main script:
```bash
python face_recognition_system.py
```

2. Controls:
- Press 'q' to quit the application
- Press 'a' to add a new face to the database
  - When prompted, enter the name for the face
  - The current frame will be used to generate embeddings

### Building Database from Images

The system includes a database builder tool that can process a folder structure of images to create a face database. The expected folder structure is:

```
base_folder/
├── IITI/
│   ├── Abhinav_Kumar.jpg
│   └── ...
├── IITB/
│   ├── Yatharth_Gupta.jpg
│   └── ...
└── IITK/
    ├── Rahul_Sharma.jpg
    └── ...
```

1. Organize your images in folders by institution
2. Run the database builder:
```bash
python create_database.py
```

The script will:
- Process all images in the institution folders
- Create standardized names (e.g., AbhinavK_IITI from Abhinav_Kumar.jpg)
- Add face embeddings to the database
- Report processing statistics

Supported image formats:
- .jpg
- .jpeg
- .png

## Configuration

### Face Recognition System

You can modify the following parameters in the `FaceRecognitionSystem` class initialization:

```python
face_system = FaceRecognitionSystem(
    database_path="face_database.pkl",  # Path to store face embeddings
    confidence_threshold=0.5,           # Minimum confidence for YOLO detection
    similarity_threshold=0.6            # Maximum distance for face recognition match
)
```

### Database Builder

You can configure the database builder in the `main()` function of `create_database.py`:

```python
base_folder = "path/to/your/base/folder"  # Path to folder containing institution folders
database_path = "face_database.pkl"        # Path to save/update the database
```

## System Architecture

The system consists of three main components:

1. **Face Detection (YOLO)**
   - Uses YOLO model optimized for face detection
   - Processes each frame to locate faces
   - Applies confidence threshold to filter detections

2. **Face Recognition (ArcFace)**
   - Generates embeddings for detected faces
   - Compares embeddings with stored database
   - Returns closest match based on similarity threshold

3. **Database Management**
   - Stores face embeddings with associated names
   - Supports adding new faces during runtime
   - Supports batch processing of image folders
   - Persists data between sessions using pickle

## Performance Considerations

- Processing speed depends on:
  - Hardware capabilities (CPU/GPU)
  - Image resolution
  - Number of faces in the database
- Adjust confidence and similarity thresholds for optimal balance between accuracy and speed

## Troubleshooting

Common issues and solutions:

1. **No camera access:**
   - Check if camera is properly connected
   - Verify camera permissions
   - Try changing camera index in `cv2.VideoCapture(0)`

2. **Poor recognition accuracy:**
   - Adjust `similarity_threshold` for stricter/looser matching
   - Ensure good lighting conditions
   - Add multiple embeddings per person under different conditions

3. **Slow performance:**
   - Reduce frame resolution
   - Increase confidence threshold to process fewer detections
   - Consider using GPU acceleration if available

4. **Database builder issues:**
   - Ensure images are in supported formats
   - Check folder structure matches expected hierarchy
   - Verify image file names follow the expected format (Name_Surname.jpg)
   - Check console output for specific error messages and processing statistics