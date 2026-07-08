import os
import requests
import base64
import json
import time
import numpy as np

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")  # DO NOT HARDCODE YOUR API KEY HERE

class VLMReasoner:
    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def _post_with_retry(self, headers, payload, max_retries=5):
        """Network request retry mechanism with exponential backoff to prevent API failures"""
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                return response
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt
                print(f"[Network] API request failed (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print("[Network] Maximum retry attempts reached. Discarding request.")
                    return None
        return None

    def encode_image(self, image):
        """Convert numpy/cv2 image to Base64 encoding"""
        import cv2
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_event(self, clip_frames, scene_prompt="", num_frames=8):
        """
        Layer 3: Dual-Track VLM Hypothesis Generator (Multi-frame Temporal Version)
        """
        print(f"[Layer 3 - VLM] Requesting dual-track hypothesis generation (using {num_frames} frames)...")
        
        # Uniformly sample frames
        total_frames = len(clip_frames)
        indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        sampled_images = [clip_frames[i] for i in indices]
        
        base64_images = [self.encode_image(img) for img in sampled_images]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Integrate user-suggested scene context prior
        context_str = ""
        if scene_prompt:
            context_str = f"[Scene Context Prior]: {scene_prompt}\n"
            
        prompt = (
            f"{context_str}"
            "[Task Description]: You are a reasoning assistant responsible for criminal investigation or video analysis. Please carefully observe the provided continuous multi-frame images (they constitute a complete video action sequence).\n"
            "Your task is to generate a [Dual-Track Hypothesis] explaining what is happening in the scene by observing the motion over time. Do NOT provide a final judgment; only list the possibilities and the required physical evidence.\n"
            "For each hypothesis, list the physical evidence (specific objects or clearly visible actions) required to prove the hypothesis, and assign a weight (MUST_HAVE, SUPPORTING, CONTEXTUAL).\n"
            "\n"
            "[Output Format (Strictly return JSON)]:\n"
            "{\n"
            "  \"H_normal\": [\n"
            "     {\n"
            "       \"desc\": \"Normal situation explanation (e.g., exercising)\",\n"
            "       \"evidence\": [\n"
            "          {\"query\": \"yoga mat\", \"type\": \"object\", \"weight\": \"MUST_HAVE\"},\n"
            "          {\"query\": \"steady posture\", \"type\": \"action\", \"weight\": \"SUPPORTING\"}\n"
            "       ]\n"
            "     }\n"
            "  ],\n"
            "  \"H_abnormal\": [\n"
            "     {\n"
            "       \"desc\": \"Abnormal situation explanation (e.g., armed assault)\",\n"
            "       \"evidence\": [\n"
            "          {\"query\": \"knife\", \"type\": \"object\", \"weight\": \"MUST_HAVE\"},\n"
            "          {\"query\": \"rapid arm swing\", \"type\": \"action\", \"weight\": \"SUPPORTING\"},\n"
            "          {\"query\": \"night or dark alley\", \"type\": \"context\", \"weight\": \"CONTEXTUAL\"}\n"
            "       ]\n"
            "     }\n"
            "  ]\n"
            "}\n"
            "\n"
            "[Ultimate Constraints]:\n"
            "1. H_normal MUST contain at least one reasonable normal daily explanation.\n"
            "2. H_abnormal MUST contain at least one abnormal or dangerous explanation.\n"
            "3. query MUST be a short, clear English phrase for the underlying CLIP/YOLO retrieval.\n"
            "4. weight can only be MUST_HAVE, SUPPORTING, or CONTEXTUAL."
        )

        content_list = []
        for b64 in base64_images:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content_list.append({
            "type": "text",
            "text": prompt
        })

        payload = {
            "model": "qwen-vl-max",
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "response_format": {"type": "json_object"}
        }

        response = self._post_with_retry(headers, payload)
        
        if response is not None and response.status_code == 200:
            result_text = response.json()['choices'][0]['message']['content']
            try:
                result_json = json.loads(result_text)
                return result_json
            except Exception as e:
                print(f"[Error] VLM JSON parsing failed: {result_text}")
                return None
        else:
            error_msg = response.text if response is not None else "Response is None due to network failure"
            print(f"[Error] VLM API request failed: {error_msg}")
            return None

    def judge_event(self, clip_frames, hypotheses, evidence_probs, scene_prompt="", num_frames=8):
        """
        Layer 5: LLM-as-a-Judge (Multi-frame Temporal Version)
        """
        print(f"[Layer 5 - Judge] Calling VLM for final judgment (using {num_frames} frames)...")
        
        # Uniformly sample frames
        total_frames = len(clip_frames)
        indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        sampled_images = [clip_frames[i] for i in indices]
        
        base64_images = [self.encode_image(img) for img in sampled_images]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Organize evidence and hypotheses
        context_str = f"[Scene Context Prior]: {scene_prompt}\n" if scene_prompt else ""
        
        hypotheses_str = json.dumps(hypotheses, ensure_ascii=False, indent=2)
        evidence_str = json.dumps(evidence_probs, ensure_ascii=False, indent=2)
        
        prompt = (
            f"{context_str}"
            "[Task Description]: You are the final judge responsible for criminal investigation. This is a continuous multi-frame temporal sequence of surveillance footage.\n"
            "By observing these continuous frames, you can clearly see the temporal evolution of the event.\n"
            "In previous analyses, we proposed the following dual-track hypotheses:\n"
            f"{hypotheses_str}\n\n"
            "Meanwhile, we deployed underlying visual detectives (YOLO and CLIP) to find evidence, and they returned the following actual existence probabilities (0~1) for objects:\n"
            f"{evidence_str}\n\n"
            "Your task: Combine the motion coherence provided by the multi-frame temporal sequence and the probability feedback above to conduct deep reflection.\n"
            "Note: You can judge causality from the continuous frames. If it is merely normal confrontation, pushing, or crowding without escalating into real violence, please assign a low score.\n"
            "Please provide a final judgment, strictly returning the following JSON format:\n"
            "{\n"
            "  \"reasoning\": \"Combining temporal frames and tool feedback, although probabilities suggest an anomaly, the continuous frames show that the two individuals only pushed briefly and then separated... therefore...\",\n"
            "  \"decision\": \"KNOWN_ANOMALY\" or \"NORMAL\",\n"
            "  \"anomaly_score\": 0.85 (Float between 0.0 and 1.0. Extremely abnormal near 1.0, completely normal near 0.0)\n"
            "}\n"
        )

        content_list = []
        for b64 in base64_images:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content_list.append({
            "type": "text",
            "text": prompt
        })

        payload = {
            "model": "qwen-vl-max",
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "response_format": {"type": "json_object"}
        }

        response = self._post_with_retry(headers, payload)
        
        if response is not None and response.status_code == 200:
            result_text = response.json()['choices'][0]['message']['content']
            try:
                result_json = json.loads(result_text)
                return result_json.get("decision", "NORMAL"), result_json.get("anomaly_score", 0.0), result_json.get("reasoning", "")
            except Exception as e:
                print(f"[Error] VLM Judge JSON parsing failed: {result_text}")
                return "NORMAL", 0.0, ""
        else:
            error_msg = response.text if response is not None else "Response is None due to network failure"
            print(f"[Error] VLM Judge API request failed: {error_msg}")
            return "NORMAL", 0.0, ""
