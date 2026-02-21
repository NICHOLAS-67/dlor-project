# Vehicle Classification Web Application

A Streamlit-based web application for classifying vehicle images into 5 categories using TensorFlow Lite.

## Vehicle Categories
- 🚌 Bus
- 🚗 Car
- 🚛 Heavy Truck
- 🚚 Light Truck
- 🏍️ Motorcycle

## Features
- Clean, user-friendly interface
- Real-time image classification
- Confidence scores for all categories
- Visual progress bars showing prediction probabilities
- Responsive design

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure your model file is present:**
Make sure `model.tflite` is in the same directory as `vehicle_classifier_app.py`

## Usage

1. **Run the application:**
```bash
streamlit run vehicle_classifier_app.py
```

2. **Use the web interface:**
   - The app will open in your default browser (usually at `http://localhost:8501`)
   - Upload a vehicle image (JPG, JPEG, or PNG)
   - Click "Classify Vehicle" button
   - View the classification results and confidence scores

## Project Structure
```
.
├── vehicle_classifier_app.py  # Main Streamlit application
├── requirements.txt           # Python dependencies
├── model.tflite              # TensorFlow Lite model (you need to add this)
└── README.md                 # This file
```

## How It Works

1. **Model Loading:** The app loads your TensorFlow Lite model on startup
2. **Image Upload:** Users upload vehicle images through the web interface
3. **Preprocessing:** Images are resized and normalized to match model requirements
4. **Prediction:** The TFLite interpreter runs inference on the preprocessed image
5. **Results Display:** Predictions are shown with confidence scores and visual indicators

## Model Requirements

The application expects a TensorFlow Lite model (`model.tflite`) that:
- Accepts image input (RGB format)
- Outputs 5 class probabilities for the vehicle categories
- Has been trained on vehicle classification data

## Troubleshooting

**Model not found:**
- Ensure `model.tflite` is in the same directory as the app

**Import errors:**
- Run `pip install -r requirements.txt` to install all dependencies

**Image processing errors:**
- Ensure uploaded images are valid JPG, JPEG, or PNG files
- Try with a different image format if issues persist

## Customization

You can customize the application by:
- Modifying `VEHICLE_CLASSES` list to match your model's output classes
- Adjusting the image preprocessing in `preprocess_image()` function
- Changing the UI layout and styling in the main function

## Dependencies

- **Streamlit:** Web application framework
- **TensorFlow:** For TFLite model inference
- **NumPy:** Numerical operations
- **Pillow (PIL):** Image processing

## License

This project is open source and available for educational and commercial use.
