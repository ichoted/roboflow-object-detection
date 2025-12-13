from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from inference_sdk import InferenceHTTPClient
import os
import shutil
import json
import cv2
import numpy as np
import base64

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize client
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="BycfIoAeYZfVFf5NFmsv"
)

def visualize_detections(image, result):
    """
    Draws bounding boxes and labels on the image based on inference result.
    Recursively searches for any dictionary containing 'x', 'y', 'width', 'height'.
    """
    print("DEBUG: Result received:", json.dumps(result, indent=2))
    
    img_draw = image.copy()
    predictions = []

    def find_predictions(data):
        if isinstance(data, dict):
            # Check if this dict itself is a prediction
            if all(k in data for k in ["x", "y", "width", "height"]):
                predictions.append(data)
            # Recurse values
            for key, value in data.items():
                find_predictions(value)
        elif isinstance(data, list):
            for item in data:
                find_predictions(item)

    find_predictions(result)
    print(f"DEBUG: Found {len(predictions)} predictions.")

    for pred in predictions:
        # Extract coordinates (center x, center y, width, height)
        # Ensure values are numbers before converting
        try:
            x, y, w, h = float(pred["x"]), float(pred["y"]), float(pred["width"]), float(pred["height"])
            
            # Convert to top-left corner
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            class_name = pred.get("class", "object")
            confidence = pred.get("confidence", 0.0)
            
            # Draw rectangle (Red: BGR = 0, 0, 255)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 20)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            # Font scale 1.5, Thickness 3
            font_scale = 10
            font_thickness = 5
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            c2 = x1 + t_size[0] + 10, y1 + t_size[1] + 10
            cv2.rectangle(img_draw, (x1, y1), c2, (0, 0, 255), -1)
            cv2.putText(img_draw, label, (x1 + 5, y1 + t_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        except (ValueError, TypeError):
            continue

    return img_draw

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "image_b64": None})

@app.post("/process", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...)):
    # Read file content
    contents = await file.read()
    
    # Save temp for inference client (sdk requires file path usually, or bytes)
    # The SDK's run_workflow for 'images' arg can support numpy array or path.
    # Let's use path to be safe and consistent with previous code.
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(contents)
        
    # Decode image for opencv
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    formatted_result = None
    image_b64 = None

    try:
        # Run workflow
        result = client.run_workflow(
            workspace_name="chote-qp4xi",
            workflow_id="detect-count-and-visualize-6",
            images={
                "image": temp_filename
            },
            use_cache=True
        )
        
        # Format result (pretty JSON)
        formatted_result = json.dumps(result, indent=2)
        
        # Visualize
        annotated_image = visualize_detections(image, result)
        
        # Save output image
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        # Use timestamp to avoid overwriting or just a unique name
        import time
        timestamp = int(time.time())
        output_filename = f"processed_{timestamp}_{file.filename}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, annotated_image)
        
        # Encode back to base64 for HTML display
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        formatted_result = f"Error: {str(e)}"
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result": formatted_result,
        "image_b64": image_b64
    })
