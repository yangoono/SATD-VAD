import numpy as np
import networkx as nx
import math
from scipy.signal import savgol_filter

class MemEventSegmenter:
    def __init__(self, fps=30, clip_weight=0.5, graph_block_size=500, time_decay=0.01, ema_window=1.0, mad_multiplier=3, min_segment_gap=2.0):
        self.fps = fps
        self.clip_weight = clip_weight
        self.graph_block_size = graph_block_size
        self.time_decay = time_decay
        self.ema_window = ema_window
        self.mad_multiplier = mad_multiplier
        self.min_segment_gap = min_segment_gap

    def build_dynamic_graph(self, visual_features, novelty_scores):
        """
        visual_features: np.array of shape (N, D) - e.g., CLIP or YOLO embeddings
        novelty_scores: np.array of shape (N,) - Memory Bank deviation scores (replacing Flow)
        """
        n = len(visual_features)
        G = nx.Graph()
        init_k = 5
        
        # Normalize visual features for cosine similarity
        clip_norms = np.linalg.norm(visual_features, axis=1, keepdims=True)
        clip_feats = visual_features / (clip_norms + 1e-6)
        
        # Ensure novelty scores are properly shaped
        novelty_scores = np.array(novelty_scores, dtype=np.float32).reshape(-1)

        print("Building Mem-Event dynamic graph...")
        for i in range(0, n, self.graph_block_size):
            i_end = min(i + self.graph_block_size, n)
            block_size_i = i_end - i
            clip_block = clip_feats[i:i_end]
            novelty_block_i = novelty_scores[i:i_end]
            
            for j in range(0, n, self.graph_block_size):
                j_end = min(j + self.graph_block_size, n)
                block_size_j = j_end - j
                
                # Visual Similarity
                clip_sim_block = np.dot(clip_block, clip_feats[j:j_end].T)
                
                # Novelty Difference (Memory Bank deviation replaces Optical Flow difference)
                novelty_block_j = novelty_scores[j:j_end]
                # Absolute difference in novelty: if two frames have similar deviation from background, they are likely the same event
                novelty_dist_block = np.abs(novelty_block_i[:, np.newaxis] - novelty_block_j)
                
                # Time Penalty
                time_diff = np.abs(np.arange(i, i_end)[:, None] - np.arange(j, j_end))
                time_penalty = 1 + self.time_decay * time_diff
                
                # Combined Similarity (EventVAD Formula, swapping flow with novelty)
                combined_sim = (self.clip_weight * clip_sim_block + 
                               (1 - self.clip_weight) * np.exp(-novelty_dist_block)) / time_penalty
                               
                dynamic_k = max(3, init_k - (i // max(1, n // 10)))
                valid_k = min(dynamic_k, block_size_j)
                
                for local_i in range(block_size_i):
                    global_i = i + local_i
                    if valid_k <= 0: continue
                    kth = min(valid_k - 1, block_size_j - 1)
                    if kth < 0: continue
                    
                    # Get top K most similar frames to connect edges
                    top_k = np.argpartition(-combined_sim[local_i], kth)[:valid_k]
                    for local_j in top_k:
                        global_j = j + local_j
                        if global_i != global_j and combined_sim[local_i, local_j] > 0:
                            G.add_edge(global_i, global_j, weight=combined_sim[local_i, local_j])

        for i in range(n):
            # Store the novelty score in the graph to compute diffs later during boundary detection
            G.add_node(i, feature=visual_features[i], novelty=novelty_scores[i])
            
        return G

    def graph_propagation(self, G, iterations=2):
        """
        EventVAD graph propagation to smooth features over the dynamic graph
        """
        n = G.number_of_nodes()
        new_G = G.copy()
        
        for _ in range(iterations):
            for i in range(n):
                neighbors = list(G.neighbors(i))
                if not neighbors:
                    continue
                weights = np.array([G[i][j]['weight'] for j in neighbors])
                sum_weights = np.sum(weights)
                if sum_weights > 0:
                    weights = weights / sum_weights
                    
                    # Propagate Visual Features
                    neighbor_features = np.array([G.nodes[j]['feature'] for j in neighbors])
                    propagated_feature = np.sum(neighbor_features * weights[:, np.newaxis], axis=0)
                    new_G.nodes[i]['feature'] = 0.5 * G.nodes[i]['feature'] + 0.5 * propagated_feature
                    
                    # Propagate Novelty
                    neighbor_novelties = np.array([G.nodes[j]['novelty'] for j in neighbors])
                    propagated_novelty = np.sum(neighbor_novelties * weights)
                    new_G.nodes[i]['novelty'] = 0.5 * G.nodes[i]['novelty'] + 0.5 * propagated_novelty
                    
        return new_G

    def detect_boundaries(self, G):
        """
        EventVAD boundary detection using Savitzky-Golay and MAD threshold
        """
        n = G.number_of_nodes()
        if n < 2: return []
        
        features = np.array([G.nodes[i]['feature'] for i in range(n)])
        novelties = np.array([G.nodes[i]['novelty'] for i in range(n)])
        
        # 1. Feature Differences
        diffs = np.diff(features, axis=0)
        s = np.linalg.norm(diffs, axis=1)**2
        
        cos_sim = np.array([
            np.dot(features[i], features[i+1]) / 
            (np.linalg.norm(features[i]) * np.linalg.norm(features[i+1]) + 1e-6)
            for i in range(n-1)
        ])
        s_cos = 1 - cos_sim
        
        # 2. Novelty Differences (replacing Flow)
        novelty_diffs = np.abs(np.diff(novelties))
        
        # Combined Signal
        s_combined = s + s_cos + novelty_diffs
        
        window_size = max(1, int(self.fps * self.ema_window))
        if len(s_combined) < window_size * 2: 
            return []
            
        # EventVAD Core: Savgol Smoothing
        window_length = max(window_size | 1, 5) # Ensure > polyorder(2) and is odd
        if len(s_combined) < window_length:
            return []
            
        s_smoothed = savgol_filter(s_combined, window_length=window_length, polyorder=2)
        
        # EMA
        ema = np.convolve(s_smoothed, np.ones(window_size)/window_size, mode='valid')
        s_ratio = s_smoothed[window_size-1:] / (ema + 1e-6)
        
        # MAD Thresholding
        median = np.median(s_ratio)
        mad = np.median(np.abs(s_ratio - median))
        threshold = median + self.mad_multiplier * mad
        
        boundaries = np.where(s_ratio > threshold)[0] + window_size // 2
        
        # Merge close boundaries
        merged = []
        prev = boundaries[0] if boundaries.size > 0 else None
        for b in boundaries[1:]:
            if prev is not None and (b - prev) < self.min_segment_gap * self.fps:
                prev = b
            else:
                if prev is not None: merged.append(prev)
                prev = b
        if prev is not None: merged.append(prev)
        
        # Return time boundaries (in seconds) or frame indices
        frame_boundaries = []
        for i in range(len(merged)):
            start_frame = merged[i]
            end_frame = (merged[i+1] if i+1 < len(merged) else min(merged[i] + int(self.min_segment_gap * self.fps), n))
            frame_boundaries.append((start_frame, end_frame))
            
        return frame_boundaries

    def process(self, visual_features, novelty_scores):
        """
        End-to-End Mem-Event Segmentation
        """
        assert len(visual_features) == len(novelty_scores)
        G = self.build_dynamic_graph(visual_features, novelty_scores)
        G = self.graph_propagation(G)
        boundaries = self.detect_boundaries(G)
        
        # Format the boundaries to cover the entire video seamlessly
        final_segments = []
        current_start = 0
        
        for b in boundaries:
            event_start, event_end = b
            if event_start > current_start:
                final_segments.append((current_start, event_start, "Normal/Background"))
            final_segments.append((event_start, event_end, "Anomaly Event"))
            current_start = event_end
            
        if current_start < len(visual_features):
            final_segments.append((current_start, len(visual_features), "Normal/Background"))
            
        # RESTORE MAX_SEG_LEN LOGIC: Split long segments into chunks of max 32 frames
        # This guarantees high-frequency scoring while maintaining multi-frame input
        MAX_SEG_LEN = 32
        chunked_segments = []
        for start_idx, end_idx, label in final_segments:
            seg_len = end_idx - start_idx
            if seg_len > MAX_SEG_LEN:
                num_chunks = math.ceil(seg_len / MAX_SEG_LEN)
                chunk_size = math.ceil(seg_len / num_chunks)
                for i in range(num_chunks):
                    c_start = start_idx + i * chunk_size
                    c_end = min(start_idx + (i + 1) * chunk_size, end_idx)
                    chunked_segments.append((c_start, c_end, label))
            else:
                chunked_segments.append((start_idx, end_idx, label))

        return chunked_segments
