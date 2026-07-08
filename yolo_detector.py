from ultralytics import YOLO
import cv2

class OpenVocabDetector:
    def __init__(self, model_id='yolov8s-world.pt'):
        """
        Initialize YOLO-World open-vocabulary detector.
        Uses yolov8s-world.pt (small model, extremely lightweight, easily runs on 8GB VRAM)
        """
        print(f"[Init] Loading visual foundation model {model_id}...")
        self.model = YOLO(model_id)
        
    def set_vocabulary(self, classes):
        """
        Dynamically modify the detector's vocabulary (core capability of our closed-loop)
        """
        # Avoid the classic Device Mismatch Bug in PyTorch/Ultralytics
        # Force model back to CPU for text CLIP encoding, then put it back to original device
        import torch
        device = next(self.model.model.parameters()).device
        self.model.to('cpu')
        
        self.model.set_classes(classes)
        
        self.model.to(device)
        print(f"[YOLO-World] Vocabulary updated to: {classes} (ready on {device})")
        
    def track(self, image_path, conf=0.1):
        """
        Enable ByteTrack temporal tracking to provide stable object IDs for probability memory
        """
        results = self.model.track(image_path, persist=True, conf=conf, save=False, verbose=False)
        return results[0]

    def detect(self, image_path, conf=0.1):
        """Single frame static detection"""
        results = self.model.predict(image_path, conf=conf, save=False, verbose=False)
        return results[0]

if __name__ == "__main__":
    # Simple initialization test
    detector = OpenVocabDetector()
    detector.set_vocabulary(["person", "bicycle", "backpack"])
    print("[Success] YOLO-World foundation initialized successfully!")
