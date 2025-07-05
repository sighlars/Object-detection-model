import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# ======= MODEL SETUP =======
# Load the standard YOLOv8 model (downloads it if not present)
model = YOLO('yolov8n.pt')  # Standard YOLOv8 model with 80 COCO classes

def detect_objects(image, confidence_threshold=0.25):
    """
    Detect objects in the uploaded image using YOLOv8
    
    Args:
        image: PIL Image or numpy array
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        annotated_image: Image with bounding boxes
        detection_summary: Text summary of detections
    """
    if image is None:
        return None, "Please upload an image first."
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array for YOLO
    img_array = np.array(image)
    
    # Run inference with YOLOv8
    results = model.predict(img_array, conf=confidence_threshold)
    
    # Get the results from the first image
    result = results[0]
    
    # Create annotated image with bounding boxes
    annotated_img = result.plot()
    
    # Convert back to PIL Image for Gradio
    annotated_img = Image.fromarray(annotated_img)
    
    # Generate detection summary
    detection_summary = generate_summary(result)
    
    return annotated_img, detection_summary

def generate_summary(result):
    """Generate a text summary of the detections"""
    if result.boxes is None or len(result.boxes) == 0:
        return "No objects detected with the current confidence threshold."
    
    # Create summary text
    summary_lines = [f"🎯 **Detected {len(result.boxes)} objects:**\n"]
    
    # Create a dictionary to count objects by class
    class_counts = {}
    detections_detail = []
    
    # Process each detection
    for i, box in enumerate(result.boxes):
        # Get class name and confidence
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        confidence = box.conf[0].item()
        
        # Add to count
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
        
        # Add detailed detection info
        detections_detail.append(f"  • {class_name}: {confidence:.2%} confidence")
    
    # Add class counts summary
    summary_lines.append("**📊 Summary by Object Type:**")
    for class_name, count in sorted(class_counts.items()):
        summary_lines.append(f"  • {class_name}: {count}")
    
    # Add detailed detections
    summary_lines.append("\n**🔍 Detailed Detections:**")
    summary_lines.extend(detections_detail)
    
    # Add model info
    summary_lines.append("\n**ℹ️ Model Information:**")
    summary_lines.append("  • Model: YOLOv8n (nano variant)")
    summary_lines.append("  • Classes: 80 COCO dataset classes")
    summary_lines.append("  • Framework: Ultralytics YOLOv8")
    
    return "\n".join(summary_lines)

# ======= GRADIO INTERFACE =======
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="🎯 Object Detection with YOLOv8") as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎯 Object Detection with YOLOv8</h1>
            <p>Upload an image to detect objects from the 80 COCO classes</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.HTML("<h3>📤 Upload Image</h3>")
                input_image = gr.Image(
                    label="Choose an image",
                    type="pil",
                    height=400
                )
                
                # Confidence threshold slider
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="🎚️ Confidence Threshold",
                    info="Minimum confidence for detections"
                )
                
                # Detect button
                detect_btn = gr.Button(
                    "🔍 Detect Objects",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                # Output section
                gr.HTML("<h3>📋 Detection Results</h3>")
                output_image = gr.Image(
                    label="Detection Results",
                    height=400
                )
                
                # Detection summary
                detection_text = gr.Textbox(
                    label="📊 Detection Summary",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
        
        # Examples section
        gr.HTML("<h3>💡 Try These Examples</h3>")
        
        # Example images (you can add your own example images here)
        examples = gr.Examples(
            examples=[
                # You can add example images here
                # ["path/to/example1.jpg", 0.25],
                # ["path/to/example2.jpg", 0.3],
            ],
            inputs=[input_image, confidence_slider],
            outputs=[output_image, detection_text],
            fn=detect_objects,
            cache_examples=False
        )
        
        # Event handlers
        detect_btn.click(
            fn=detect_objects,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, detection_text]
        )
        
        # Auto-detect when image is uploaded
        input_image.change(
            fn=detect_objects,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, detection_text]
        )
        
        # Auto-detect when confidence threshold changes
        confidence_slider.change(
            fn=detect_objects,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, detection_text]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>Powered by YOLOv8 and Gradio | Detects 80 COCO classes including people, animals, vehicles, and everyday objects</p>
        </div>
        """)
    
    return interface

# ======= LAUNCH APP =======
if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    
    # Launch options
    interface.launch(
        server_name="127.0.0.1",  # Make accessible from any IP
        server_port=7860,       # Default Gradio port
        share=True,            # Create public sharing link
        show_error=True,       # Show errors in interface
        pwa=True               # Enable PWA mode
    )