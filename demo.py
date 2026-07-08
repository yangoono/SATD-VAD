import os
import cv2
import json
import torch
import numpy as np
from PIL import Image

from mem_event_segmenter import MemEventSegmenter
from vlm_reasoner import VLMReasoner
from evidence_retriever import EvidenceRetriever
from yolo_detector import OpenVocabDetector
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_clip_features_batch(frames, clip_model, clip_processor, batch_size=128):
    """Extract CLIP features in batches to prevent OOM"""
    all_features = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        pil_images = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB)) for f in batch_frames]
        inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy())
    if len(all_features) == 0:
        return np.array([])
    return np.vstack(all_features)

def compute_novelty_scores(features, init_normal_frames=5):
    """
    Compute Memory Bank Deviation (Replacing Optical Flow).
    Assumes the first N frames are normal background, and computes cosine distance as deviation.
    """
    if len(features) == 0:
        return np.array([])
    
    # Build initial prototype memory
    memory_bank = features[:init_normal_frames]
    
    novelty_scores = []
    for f in features:
        # Calculate distance with all prototypes in memory bank, take the most similar
        sims = np.dot(memory_bank, f)
        max_sim = np.max(sims)
        novelty = 1.0 - max_sim # Deviation = 1 - Similarity
        novelty_scores.append(novelty)
        
        # Self-evolving write: if anomaly is extremely low, fuse it into memory
        if novelty < 0.05 and len(memory_bank) < 50:
            memory_bank = np.vstack([memory_bank, f])
            
    return np.array(novelty_scores)

def load_video_frames(video_path, sample_rate=16):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_indices = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frames.append(frame)
            frame_indices.append(count)
        count += 1
    cap.release()
    return frames, frame_indices, count

def main():
    print("[*] SATD Zero-Shot Single Video Inference Demo")
    
    video_path = "sample_video.mp4" # Replace with your video path
    if not os.path.exists(video_path):
        print(f"[!] Video not found: {video_path}. Please provide a valid video path.")
        return
        
    print("[*] Loading Models (VLM, YOLO-World, CLIP)...")
    detector = OpenVocabDetector('yolov8s-world.pt')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    reasoner = VLMReasoner()
    retriever = EvidenceRetriever(detector, clip_model, clip_processor, device)
    
    # Initialize Stage 1 core: MemEventSegmenter
    segmenter = MemEventSegmenter(fps=30//16, mad_multiplier=2.5)

    print(f"\n>>>> [Processing] Video: {video_path}")
    
    # --- Stage 1: Feature Extraction & Dynamic Graph Slicing ---
    print("[Stage 1] Extracting frames and features...")
    sampled_frames, frame_indices, total_frames = load_video_frames(video_path, sample_rate=16)
    
    if len(sampled_frames) < 10:
        print("  - Video is too short, evaluating as a single segment.")
        features = get_clip_features_batch(sampled_frames, clip_model, clip_processor)
        segments = [(0, len(sampled_frames)-1 if len(sampled_frames)>0 else 0, "Short Anomaly")]
    else:
        features = get_clip_features_batch(sampled_frames, clip_model, clip_processor)
        novelty_scores = compute_novelty_scores(features)
        segments = segmenter.process(features, novelty_scores)
        
    final_scores = np.zeros(len(sampled_frames))
    
    for start_idx, end_idx, label in segments:
        if (end_idx - start_idx) > 0:
            print(f"\n[Stage 1] Evaluating Segment ({label}): {start_idx} -> {end_idx}")
            
            mid_idx = (start_idx + end_idx) // 2
            key_frame = sampled_frames[mid_idx]
            clip_frames = sampled_frames[start_idx:end_idx]
            
            # --- Stage 2: VLM Hypothesis ---
            print("  [Stage 2] VLM is generating hypothesis and retrieving vocabulary...")
            vlm_json = reasoner.analyze_event(clip_frames, scene_prompt="", num_frames=8)
            
            if vlm_json:
                all_evidence = []
                for h in vlm_json.get("H_normal", []):
                    all_evidence.extend(h.get("evidence", []))
                for h in vlm_json.get("H_abnormal", []):
                    all_evidence.extend(h.get("evidence", []))
                    
                # --- Stage 3: CLIP/YOLO Grounding ---
                print("  [Stage 3] YOLO & CLIP are searching for objective evidence...")
                event_clip_dict = {
                    'key_frame': key_frame,
                    'clip_frames': clip_frames
                }
                evidence_probs = retriever.retrieve(all_evidence, event_clip_dict)
                
                # --- Stage 4: VLM Judge ---
                print("  [Stage 4] VLM is making the final judgement based on evidence...")
                decision, score, reasoning = reasoner.judge_event(clip_frames, vlm_json, evidence_probs, scene_prompt="", num_frames=8)
                print(f"    -> Decision: {decision}, Score: {score:.4f}")
                print(f"    -> Reasoning: {reasoning}")
                
                final_scores[start_idx:end_idx] = score
            else:
                print("  - VLM failed to respond, skipping segment.")
    
    print("\n[*] Inference Complete. Final Scores:")
    print(final_scores)

if __name__ == "__main__":
    main()
