import numpy as np
import torch

class EvidenceRetriever:
    """
    Layer 4: Hierarchical Probability Evidence Retrieval
    Convert text evidence into continuous probabilities.
    Supports Object (based on YOLO-World) and Action/Context (based on CLIP + local smoothing).
    """
    def __init__(self, detector, clip_model, clip_processor, device="cpu"):
        self.detector = detector
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device
        
    def _get_object_prob(self, query, key_frame):
        """
        Object-level evidence: detected using YOLO-World.
        Here we temporarily set the vocabulary to the query and take the maximum confidence returned as P(e_obj).
        """
        self.detector.set_vocabulary([query])
        # Use extremely low confidence threshold to let the model output possibilities
        results = self.detector.detect(key_frame, conf=0.01)
        
        max_conf = 0.001 # Base minimum probability to prevent log(0)
        
        if results.boxes is not None and len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            if len(confs) > 0:
                max_conf = float(np.max(confs))
                
        # Restoring default vocabulary can be controlled externally
        return max_conf

    def _get_action_prob(self, query, event_clip_frames):
        """
        Action/Relation-level evidence: using CLIP Text-Image similarity.
        Perform sliding window or average pooling smoothing similar to CoReVAD within the event segment.
        """
        from PIL import Image
        import cv2
        
        if not event_clip_frames:
            return 0.001
            
        # Convert opencv format to PIL
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in event_clip_frames]
        
        inputs = self.clip_processor(text=[query], images=pil_frames, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # Image-text similarity score
            probs = logits_per_image.softmax(dim=1) # Not strictly category probability, just for normalization reference
            
            # Use simple cosine similarity and normalize to [0, 1] as approximate probability
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # shape: (num_frames, 1)
            similarities = (image_embeds @ text_embeds.T).squeeze()
            
            if len(similarities.shape) == 0:
                similarities = similarities.unsqueeze(0)
                
            sim_np = similarities.cpu().numpy()
            
        # [Core Fix: Resolve probability calibration imbalance between CLIP and YOLO]
        # CLIP's text-image cosine similarity is usually pure background noise (completely irrelevant) between 0.2~0.25
        # Only above 0.28~0.30 does it represent the true existence of the relevant action.
        # The previous (sim+1)/2 would map 0.25 noise to a high probability of 0.625, causing abnormal actions to be frequently misjudged as existing!
        # We use a Sigmoid function to map 0.25 close to 0, and 0.30 close to 1.
        bias = 0.26  # Set 0.26 as the true/false boundary (can be fine-tuned based on actual model)
        scale = 40.0 # Steepness
        
        # Sigmoid mapping
        mapped_probs = 1.0 / (1.0 + np.exp(-scale * (sim_np - bias)))
        
        # Limit to [0.001, 0.99]
        mapped_probs = np.clip(mapped_probs, 0.001, 0.99)
        
        # Temporal smoothing: Take average of Top-K or direct max, simulating CoReVAD LRC
        # Here for demonstration, take 90th percentile, representing the peak moment of the action
        p_act = float(np.percentile(mapped_probs, 90))
        return p_act

    def retrieve(self, evidence_list, event_clip):
        """
        Process the list of evidence returned by the VLM.
        Returns a dictionary of evidence probabilities.
        """
        print(f"[Layer 4 - EvidenceRetriever] Starting probabilistic evidence retrieval...")
        probs_dict = {}
        key_frame = event_clip['key_frame']
        clip_frames = event_clip.get('clip_frames', [key_frame])
        
        object_queries = []
        for ev in evidence_list:
            if ev.get('type', 'object') == 'object' and ev.get('query'):
                object_queries.append(ev['query'])
                
        if object_queries:
            self.detector.set_vocabulary(object_queries)
            results = self.detector.detect(key_frame, conf=0.01)
            
            for q in object_queries:
                probs_dict[q] = 0.001
                
            if results.boxes is not None and len(results.boxes) > 0:
                confs = results.boxes.conf.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy().astype(int)
                
                for c_id, conf in zip(clss, confs):
                    if c_id < len(object_queries):
                        q = object_queries[c_id]
                        if conf > probs_dict[q]:
                            probs_dict[q] = float(conf)
        
        for ev in evidence_list:
            query = ev.get('query', '')
            ev_type = ev.get('type', 'object')
            
            if not query:
                continue
                
            if ev_type == 'object':
                p = probs_dict.get(query, 0.001)
            else:
                p = self._get_action_prob(query, clip_frames)
                probs_dict[query] = p
                
            print(f"  - Evidence [{query}] ({ev_type}): P = {p:.4f}")
            
        return probs_dict
