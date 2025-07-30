import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import filters, measure
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from typing import Dict, Tuple, Optional
import json
import torch
import torch.nn.functional as F
import pytesseract
import easyocr
import re

class OCREvaluationTool:
    """
    A comprehensive OCR evaluation tool that analyzes image quality metrics
    relevant for Optical Character Recognition performance.
    """
    
    def __init__(self, debug=False, matching_mode="text_aware", levenshtein_threshold=1):
        """
        Initialize the OCR evaluation tool.
        
        Args:
            debug: Enable debug output
            matching_mode: Matching strategy mode
                - "text_aware": Prioritize text similarity with position as secondary (default)
                - "position_focused": Prioritize bounding box similarity with text as secondary
            levenshtein_threshold: Maximum Levenshtein distance allowed for word matching (default: 1)
        """
        self.metrics = {}
        self.debug = debug
        self.matching_mode = matching_mode
        self.levenshtein_threshold = levenshtein_threshold
        
        # Validate matching mode
        if matching_mode not in ["text_aware", "position_focused"]:
            raise ValueError(f"Invalid matching_mode: {matching_mode}. Must be 'text_aware' or 'position_focused'")
        
        if self.debug:
            print(f"🎯 Matching Mode: {matching_mode}")
            print(f"🔍 Levenshtein Threshold: {levenshtein_threshold}")
        
        # Check for CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        
        if self.use_cuda:
            print(f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 Using CPU for computations")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for consistency
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for some metrics
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image_rgb, image_gray
    
    def calculate_laplacian_variance(self, image_gray: np.ndarray) -> float:
        """
        Calculate Laplacian variance - a measure of image sharpness.
        Higher values indicate sharper images.
        
        Args:
            image_gray: Grayscale image
            
        Returns:
            Laplacian variance value
        """
        if self.use_cuda:
            # Use PyTorch for GPU acceleration
            image_tensor = torch.from_numpy(image_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Laplacian kernel
            laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=self.device)
            
            # Apply convolution
            laplacian = F.conv2d(image_tensor, laplacian_kernel, padding=1)
            
            # Calculate variance
            laplacian_variance = torch.var(laplacian).item()
        else:
            # Use OpenCV for CPU
            laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
            laplacian_variance = np.var(laplacian)
        
        return laplacian_variance
    
    def calculate_contrast(self, image_gray: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation of pixel values.
        
        Args:
            image_gray: Grayscale image
            
        Returns:
            Contrast value
        """
        # Calculate standard deviation as a measure of contrast
        contrast = np.std(image_gray)
        
        return contrast
    
    def calculate_noise_estimation(self, image_gray: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation.
        
        Args:
            image_gray: Grayscale image
            
        Returns:
            Noise estimation value
        """
        if self.use_cuda:
            # Use PyTorch for GPU acceleration
            image_tensor = torch.from_numpy(image_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Gaussian blur kernel (5x5)
            gaussian_kernel = torch.tensor([
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ], dtype=torch.float32, device=self.device) / 256.0
            
            gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
            
            # Apply Gaussian blur
            blurred = F.conv2d(image_tensor, gaussian_kernel, padding=2)
            
            # Calculate difference
            noise = image_tensor - blurred
            
            # Calculate median absolute deviation
            noise_estimate = torch.median(torch.abs(noise)).item()
        else:
            # Use OpenCV for CPU
            blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
            noise = image_gray.astype(np.float64) - blurred.astype(np.float64)
            noise_estimate = np.median(np.abs(noise))
        
        return noise_estimate
    
    def calculate_structural_similarity_index(self, image_gray: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) using a reference image.
        Since we don't have a reference, we'll use a blurred version as reference.
        
        Args:
            image_gray: Grayscale image
            
        Returns:
            SSIM value
        """
        # Create a reference image (blurred version)
        reference = cv2.GaussianBlur(image_gray, (15, 15), 0)
        
        # Calculate SSIM
        ssim_value = ssim(image_gray, reference, data_range=255)
        
        return ssim_value
    
    def calculate_multiscale_ssim(self, image_gray: np.ndarray) -> float:
        """
        Calculate Multi-scale Structural Similarity Index (MS-SSIM).
        
        Args:
            image_gray: Grayscale image
            
        Returns:
            MS-SSIM value
        """
        # Create reference image (blurred version)
        reference = cv2.GaussianBlur(image_gray, (15, 15), 0)
        
        # Calculate MS-SSIM with multiple scales
        # We'll simulate this by calculating SSIM at different resolutions
        scales = [1.0, 0.5, 0.25]
        msssim_values = []
        
        for scale in scales:
            if scale != 1.0:
                # Resize images
                h, w = image_gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                
                img_scaled = cv2.resize(image_gray, (new_w, new_h))
                ref_scaled = cv2.resize(reference, (new_w, new_h))
                
                ssim_scaled = ssim(img_scaled, ref_scaled, data_range=255)
            else:
                ssim_scaled = ssim(image_gray, reference, data_range=255)
            
            msssim_values.append(ssim_scaled)
        
        # Calculate weighted average (giving more weight to higher resolution)
        weights = [0.5, 0.3, 0.2]  # Weights for different scales
        msssim = np.average(msssim_values, weights=weights)
        
        return msssim
    
    def _levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Levenshtein distance
        """
        # Create a matrix of size (len(str1)+1) x (len(str2)+1)
        matrix = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
        
        # Initialize first row and column
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j
        
        # Fill the matrix
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i-1] == str2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,    # deletion
                        matrix[i][j-1] + 1,    # insertion
                        matrix[i-1][j-1] + 1   # substitution
                    )
        
        return matrix[len(str1)][len(str2)]
    
    def _calculate_word_level_metrics(self, detected: str, ground_truth: str) -> dict:
        """Calculate word-level Levenshtein distance and accuracy metrics with fuzzy matching."""
        detected_words = detected.lower().split()
        gt_words = ground_truth.lower().split()
        
        # Handle empty cases
        if not detected_words and not gt_words:
            return {
                'word_levenshtein_distance': 0.0,
                'word_accuracy': 1.0,
                'word_precision': 1.0,
                'word_recall': 1.0,
                'detected_word_count': 0,
                'ground_truth_word_count': 0
            }
        
        if not detected_words:
            return {
                'word_levenshtein_distance': 1.0,
                'word_accuracy': 0.0,
                'word_precision': 0.0,
                'word_recall': 0.0,
                'detected_word_count': 0,
                'ground_truth_word_count': len(gt_words)
            }
        
        if not gt_words:
            return {
                'word_levenshtein_distance': 1.0,
                'word_accuracy': 0.0,
                'word_precision': 0.0,
                'word_recall': 0.0,
                'detected_word_count': len(detected_words),
                'ground_truth_word_count': 0
            }
        
        # Calculate word-level Levenshtein distance
        word_lev_dist = self._levenshtein_distance(' '.join(detected_words), ' '.join(gt_words))
        max_word_len = max(len(detected_words), len(gt_words))
        normalized_word_lev = word_lev_dist / max_word_len if max_word_len > 0 else 0.0

        # Word accuracy: proportion of matches using Levenshtein distance threshold
        matches = 0
        matched_gt_indices = set()
        matched_detected_indices = set()
        
        # For each ground truth word, find the best matching detected word
        for i, gt_word in enumerate(gt_words):
            best_match_distance = float('inf')
            best_match_index = -1
            
            for j, detected_word in enumerate(detected_words):
                if j in matched_detected_indices:
                    continue  # Skip already matched detected words
                
                distance = self._levenshtein_distance(gt_word, detected_word)
                if distance < best_match_distance and distance <= self.levenshtein_threshold:
                    best_match_distance = distance
                    best_match_index = j
            
            if best_match_index != -1:
                matches += 1
                matched_gt_indices.add(i)
                matched_detected_indices.add(best_match_index)
        
        # Calculate accuracy based on matches
        word_accuracy = matches / len(gt_words) if gt_words else 0.0

        # Precision/Recall using Levenshtein distance matching
        # Precision: how many of the detected words match ground truth words
        precision_matches = 0
        for j, detected_word in enumerate(detected_words):
            best_match_distance = float('inf')
            for gt_word in gt_words:
                distance = self._levenshtein_distance(detected_word, gt_word)
                if distance < best_match_distance and distance <= self.levenshtein_threshold:
                    best_match_distance = distance
            
            if best_match_distance <= self.levenshtein_threshold:
                precision_matches += 1
        
        precision = precision_matches / len(detected_words) if detected_words else 0.0
        recall = matches / len(gt_words) if gt_words else 0.0

        # Calculate F1 score
        word_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Debug word-level metrics
        if hasattr(self, 'debug') and self.debug:
            print(f"   Debug - Detected words: {detected_words}")
            print(f"   Debug - GT words: {gt_words}")
            print(f"   Debug - Levenshtein threshold: {self.levenshtein_threshold}")
            print(f"   Debug - Word matches: {matches}/{len(gt_words)}")
            print(f"   Debug - Word precision: {precision:.3f}, recall: {recall:.3f}")
            print(f"   Debug - Word accuracy: {word_accuracy:.3f}")
            print(f"   Debug - Word F1: {word_f1:.3f}")
        
        return {
            'word_levenshtein_distance': normalized_word_lev,
            'word_accuracy': word_accuracy,
            'word_precision': precision,
            'word_recall': recall,
            'word_f1': word_f1,
            'detected_word_count': len(detected_words),
            'ground_truth_word_count': len(gt_words)
        }
    
    def extract_bbox_regions(self, image_gray: np.ndarray, ground_truth_bboxes: list) -> list:
        """
        Extract image regions based on bounding boxes.
        
        Args:
            image_gray: Grayscale image
            ground_truth_bboxes: List of bounding boxes [x, y, width, height] in percentage
            
        Returns:
            List of extracted image regions
        """
        regions = []
        h, w = image_gray.shape
        
        for bbox in ground_truth_bboxes:
            # Convert percentage coordinates to pixel coordinates
            x_percent, y_percent, width_percent, height_percent = bbox
            
            if hasattr(self, 'debug') and self.debug:
                print(f"   Debug - Processing bbox: {bbox} (x={x_percent}%, y={y_percent}%, w={width_percent}%, h={height_percent}%)")
            
            # Validate percentage coordinates first
            if (x_percent < 0 or y_percent < 0 or width_percent <= 0 or height_percent <= 0 or
                x_percent + width_percent > 100 or y_percent + height_percent > 100):
                # Skip invalid bounding boxes
                if hasattr(self, 'debug') and self.debug:
                    print(f"   Debug - Skipping invalid bbox: {bbox} (x={x_percent}, y={y_percent}, w={width_percent}, h={height_percent})")
                continue
            
            # Convert to pixel coordinates
            x = int(x_percent * w / 100)
            y = int(y_percent * h / 100)
            width = int(width_percent * w / 100)
            height = int(height_percent * h / 100)
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = min(width, w - x)
            height = min(height, h - y)
            
            if width > 0 and height > 0:
                # Extract region
                region = image_gray[y:y+height, x:x+width]
                regions.append(region)
        
        return regions

    def visualize_bbox_regions(self, image_gray: np.ndarray, ground_truth_bboxes: list, ground_truth_texts: list = None, ocr_results: list = None, output_path: str = None):
        """
        Visualize the bounding box regions with ground truth and OCR results.
        
        Args:
            image_gray: Grayscale image
            ground_truth_bboxes: List of bounding boxes
            ground_truth_texts: List of ground truth text for each bounding box
            ocr_results: List of OCR results for each bounding box
            output_path: Optional path to save visualization
        """
        if not ground_truth_bboxes:
            return
        
        # Convert grayscale to RGB for visualization
        if len(image_gray.shape) == 2:
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_gray.copy()
        
        h, w = image_rgb.shape[:2]
        
        # Initialize EasyOCR reader for OCR processing
        reader = None
        if ocr_results is None:
            try:
                reader = easyocr.Reader(['en'], gpu=self.use_cuda)
            except Exception as e:
                if self.debug:
                    print(f"   Debug - Could not initialize OCR reader: {e}")
        
        # Draw bounding boxes with text
        for i, bbox in enumerate(ground_truth_bboxes):
            x_percent, y_percent, width_percent, height_percent = bbox
            
            # Convert to pixel coordinates
            x = int(x_percent * w / 100)
            y = int(y_percent * h / 100)
            width = int(width_percent * w / 100)
            height = int(height_percent * h / 100)
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = min(width, w - x)
            height = min(height, h - y)
            
            if width > 0 and height > 0:
                # Draw rectangle
                cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Get ground truth text
                gt_text = "No GT"
                if ground_truth_texts and i < len(ground_truth_texts):
                    gt_text = ground_truth_texts[i]
                
                # Get OCR detected text
                ocr_text = "No OCR"
                if ocr_results and i < len(ocr_results):
                    ocr_text = ocr_results[i]
                elif reader:
                    # Extract region and perform OCR
                    region = image_gray[y:y+height, x:x+width]
                    try:
                        region_results = reader.readtext(region)
                        if region_results:
                            detected_texts = [text for _, text, _ in region_results]
                            ocr_text = " | ".join(detected_texts)
                        else:
                            ocr_text = "No text detected"
                    except Exception as e:
                        ocr_text = f"OCR error: {str(e)[:20]}"
                
                # Truncate long text for display
                gt_text_display = gt_text[:30] + "..." if len(gt_text) > 30 else gt_text
                ocr_text_display = ocr_text[:30] + "..." if len(ocr_text) > 30 else ocr_text
                
                # Add ground truth text above bounding box (green) - INCREASED FONT SIZE
                cv2.putText(image_rgb, f'GT: {gt_text_display}', (x, y-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Add region number - INCREASED FONT SIZE
                cv2.putText(image_rgb, f'Region {i+1}', (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Add OCR detected text below bounding box (blue) - INCREASED FONT SIZE
                cv2.putText(image_rgb, f'OCR: {ocr_text_display}', (x, y+height+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                
                # Add confidence if available - INCREASED FONT SIZE
                if reader and ocr_text != "No OCR" and ocr_text != "No text detected":
                    try:
                        region = image_gray[y:y+height, x:x+width]
                        region_results = reader.readtext(region)
                        if region_results:
                            confidences = [conf for _, _, conf in region_results]
                            avg_conf = sum(confidences) / len(confidences)
                            cv2.putText(image_rgb, f'Conf: {avg_conf:.2f}', (x, y+height+45), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    except:
                        pass
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            if self.debug:
                print(f"   Debug - Enhanced bounding box visualization saved to: {output_path}")
        else:
            plt.figure(figsize=(15, 10))
            plt.imshow(image_rgb)
            plt.title('Ground Truth vs OCR Results')
            plt.axis('off')
            plt.show()

    def calculate_levenshtein_distance(self, image_gray: np.ndarray, ground_truth_text: str = None, ground_truth_bboxes: list = None) -> Dict[str, float]:
        """
        Calculate Levenshtein distance between OCR predictions and ground truth.
        Uses both character-level and word-level analysis for comprehensive evaluation.
        
        Args:
            image_gray: Grayscale image
            ground_truth_text: Ground truth text (optional)
            ground_truth_bboxes: List of ground truth bounding boxes (optional)
            
        Returns:
            Dictionary containing Levenshtein distance and related metrics
        """
        try:
            # Initialize EasyOCR reader (use GPU if available)
            device = 'cuda' if self.use_cuda else 'cpu'
            reader = easyocr.Reader(['en'], gpu=self.use_cuda)
            
            # Perform OCR on specific regions if bounding boxes are provided
            if ground_truth_bboxes and len(ground_truth_bboxes) > 0:
                if self.debug:
                    print(f"   Debug - Processing {len(ground_truth_bboxes)} bounding box regions")
                
                # Extract regions from bounding boxes
                regions = self.extract_bbox_regions(image_gray, ground_truth_bboxes)
                
                if not regions:
                    if self.debug:
                        print("   Debug - No valid regions extracted from bounding boxes")
                    return {
                        'levenshtein_distance': 1.0,
                        'word_levenshtein_distance': 1.0,
                        'word_accuracy': 0.0,
                        'ocr_confidence': 0.0,
                        'detected_text_count': 0,
                        'detected_word_count': 0,
                        'text_detection_quality': 0.0
                    }
                
                # Perform OCR on each region
                all_results = []
                for i, region in enumerate(regions):
                    if self.debug:
                        print(f"   Debug - Processing region {i+1}/{len(regions)}")
                    
                    region_results = reader.readtext(region)
                    
                    if self.debug and region_results:
                        region_texts = [text for _, text, _ in region_results]
                        region_confidences = [conf for _, _, conf in region_results]
                        print(f"   Debug - Region {i+1} detected: {region_texts}")
                        print(f"   Debug - Region {i+1} confidences: {[f'{c:.3f}' for c in region_confidences]}")
                    elif self.debug:
                        print(f"   Debug - Region {i+1} detected: No text")
                    
                    all_results.extend(region_results)
                
                results = all_results
            else:
                # Fallback to full image OCR if no bounding boxes provided
                if self.debug:
                    print("   Debug - No bounding boxes provided, processing entire image")
                results = reader.readtext(image_gray)
            
            if not results:
                return {
                    'levenshtein_distance': 0.0,
                    'word_levenshtein_distance': 0.0,
                    'word_accuracy': 0.0,
                    'ocr_confidence': 0.0,
                    'detected_text_count': 0,
                    'detected_word_count': 0,
                    'text_detection_quality': 0.0
                }
            
            # Extract detected text and confidence scores
            detected_texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                # Clean the detected text - preserve hyphens, periods, and spaces
                # Remove only truly problematic characters while keeping military unit format
                cleaned_text = re.sub(r'[^a-zA-Z0-9\s\-\.]', '', text).strip()
                if cleaned_text:
                    detected_texts.append(cleaned_text)
                    confidences.append(confidence)
            
            if not detected_texts:
                return {
                    'levenshtein_distance': 0.0,
                    'word_levenshtein_distance': 0.0,
                    'word_accuracy': 0.0,
                    'ocr_confidence': 0.0,
                    'detected_text_count': 0,
                    'detected_word_count': 0,
                    'text_detection_quality': 0.0
                }
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences)
            
            # Calculate Levenshtein distance if ground truth is provided
            if ground_truth_text:
                combined_detected = ' '.join(detected_texts)
                # Character-level Levenshtein
                char_lev = self._levenshtein_distance(combined_detected.lower(), ground_truth_text.lower())
                max_char_len = max(len(combined_detected), len(ground_truth_text))
                normalized_char_lev = char_lev / max_char_len if max_char_len > 0 else 0.0
                # Character Error Rate (CER)
                cer = char_lev / len(ground_truth_text) if len(ground_truth_text) > 0 else 0.0

                # Word-level metrics
                word_metrics = self._calculate_word_level_metrics(combined_detected, ground_truth_text)
                # Word Error Rate (WER) - use the word-level Levenshtein from word_metrics
                gt_word_count = word_metrics['ground_truth_word_count']
                wer = word_metrics['word_levenshtein_distance'] if gt_word_count > 0 else 0.0

                # Character-level Recall - Fixed calculation
                detected_chars = list(combined_detected.lower())
                gt_chars = list(ground_truth_text.lower())
                
                # Count character matches more accurately
                char_matches = 0
                detected_char_counts = {}
                gt_char_counts = {}
                
                # Count characters in detected text
                for char in detected_chars:
                    detected_char_counts[char] = detected_char_counts.get(char, 0) + 1
                
                # Count characters in ground truth
                for char in gt_chars:
                    gt_char_counts[char] = gt_char_counts.get(char, 0) + 1
                
                # Count matches (minimum of detected and ground truth counts)
                for char in gt_char_counts:
                    detected_count = detected_char_counts.get(char, 0)
                    gt_count = gt_char_counts[char]
                    char_matches += min(detected_count, gt_count)
                
                char_recall = char_matches / len(gt_chars) if len(gt_chars) > 0 else 0.0

                # Word-level F1 score
                precision = word_metrics['word_precision']
                recall = word_metrics['word_recall']
                word_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                # Debug information
                if self.debug:
                    print(f"   Debug - Ground Truth: '{ground_truth_text}'")
                    print(f"   Debug - Detected: '{combined_detected}'")
                    print(f"   Debug - Char Levenshtein: {char_lev}, Normalized: {normalized_char_lev:.3f}")
                    print(f"   Debug - CER: {cer:.3f}, Char Recall: {char_recall:.3f}")
                    print(f"   Debug - Word Accuracy: {word_metrics['word_accuracy']:.3f}, WER: {wer:.3f}")
                    print(f"   Debug - Word F1: {word_f1:.3f}")
                
                return {
                    'levenshtein_distance': normalized_char_lev,
                    'character_error_rate': cer,
                    'character_recall': char_recall,
                    'word_levenshtein_distance': word_metrics['word_levenshtein_distance'],
                    'word_error_rate': wer,
                    'word_accuracy': word_metrics['word_accuracy'],
                    'word_precision': word_metrics['word_precision'],
                    'word_recall': word_metrics['word_recall'],
                    'word_f1': word_f1,
                    'ocr_confidence': avg_confidence,
                    'detected_text_count': len(detected_texts),
                    'detected_word_count': word_metrics['detected_word_count'],
                    'ground_truth_word_count': word_metrics['ground_truth_word_count'],
                    'text_detection_quality': (1.0 - normalized_char_lev + word_metrics['word_accuracy']) / 2.0,
                    'detected_text': combined_detected
                }
            else:
                # Use confidence as a proxy for text detection quality
                text_detection_quality = avg_confidence
                combined_detected = ' '.join(detected_texts)
                return {
                    'levenshtein_distance': 1.0 - avg_confidence,  # Proxy distance
                    'word_levenshtein_distance': 1.0 - avg_confidence,  # Proxy distance
                    'word_accuracy': avg_confidence,  # Proxy accuracy
                    'ocr_confidence': avg_confidence,
                    'detected_text_count': len(detected_texts),
                    'detected_word_count': len(combined_detected.split()),
                    'text_detection_quality': text_detection_quality,
                    'detected_text': combined_detected
                }
                
        except Exception as e:
            print(f"Warning: OCR processing failed: {e}")
            return {
                'levenshtein_distance': 1.0,
                'word_levenshtein_distance': 1.0,
                'word_accuracy': 0.0,
                'ocr_confidence': 0.0,
                'detected_text_count': 0,
                'detected_word_count': 0,
                'text_detection_quality': 0.0
            }
    
    def calculate_additional_metrics(self, image_gray: np.ndarray) -> Dict[str, float]:
        """
        Calculate additional useful metrics for OCR evaluation.
        
        Args:
            image_gray: Grayscale image
            
        Returns:
            Dictionary of additional metrics
        """
        # Edge density
        edges = cv2.Canny(image_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Local binary pattern variance (texture measure)
        lbp = self._calculate_lbp(image_gray)
        lbp_variance = np.var(lbp)
        
        # Entropy (information content)
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            'edge_density': edge_density,
            'lbp_variance': lbp_variance,
            'entropy': entropy
        }
    
    def _calculate_lbp(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Calculate Local Binary Pattern for texture analysis.
        
        Args:
            image_gray: Grayscale image
            
        Returns:
            LBP image
        """
        # Simple LBP implementation
        lbp = np.zeros_like(image_gray)
        
        for i in range(1, image_gray.shape[0] - 1):
            for j in range(1, image_gray.shape[1] - 1):
                center = image_gray[i, j]
                code = 0
                
                # Check 8 neighbors
                neighbors = [
                    image_gray[i-1, j-1], image_gray[i-1, j], image_gray[i-1, j+1],
                    image_gray[i, j+1], image_gray[i+1, j+1], image_gray[i+1, j],
                    image_gray[i+1, j-1], image_gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def analyze_regional_ocr(self, image_gray: np.ndarray, ground_truth_bboxes: list) -> Dict[str, float]:
        """
        Analyze OCR performance in specific regions defined by bounding boxes.
        
        Args:
            image_gray: Grayscale image
            ground_truth_bboxes: List of ground truth bounding boxes
            
        Returns:
            Dictionary containing regional OCR analysis metrics
        """
        if not ground_truth_bboxes:
            return {}
        
        try:
            # Initialize EasyOCR reader
            device = 'cuda' if self.use_cuda else 'cpu'
            reader = easyocr.Reader(['en'], gpu=self.use_cuda)
            
            regional_metrics = {
                'total_regions': len(ground_truth_bboxes),
                'regions_with_text': 0,
                'average_confidence_per_region': 0.0,
                'region_detection_accuracy': 0.0
            }
            
            total_confidence = 0.0
            regions_with_text = 0
            
            # Extract regions using the same method as main OCR
            regions = self.extract_bbox_regions(image_gray, ground_truth_bboxes)
            
            for i, region in enumerate(regions):
                # Perform OCR on region
                results = reader.readtext(region)
                
                if results:
                    regions_with_text += 1
                    # Calculate average confidence for this region
                    region_confidence = np.mean([conf for _, _, conf in results])
                    total_confidence += region_confidence
            
            regional_metrics['regions_with_text'] = regions_with_text
            regional_metrics['average_confidence_per_region'] = total_confidence / len(ground_truth_bboxes) if ground_truth_bboxes else 0.0
            regional_metrics['region_detection_accuracy'] = regions_with_text / len(ground_truth_bboxes) if ground_truth_bboxes else 0.0
            
            return regional_metrics
            
        except Exception as e:
            print(f"Warning: Regional OCR analysis failed: {e}")
            return {}
    

    def create_ocr_detected_map(self, image_gray: np.ndarray, image_path: str) -> Dict[str, list]:
        """
        Run OCR on the entire image and create a map of detected text with bounding boxes.
        
        Args:
            image_gray: Grayscale image
            image_path: Path to the original image
            
        Returns:
            Dictionary containing detected text regions with bounding boxes
        """
        try:
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], gpu=self.use_cuda)
            
            if self.debug:
                print(f"   Debug - Running OCR on entire image to detect text regions...")
            
            # Run OCR on the entire image
            results = reader.readtext(image_gray)
            
            if self.debug:
                print(f"   Debug - Detected {len(results)} text regions")
            
            # Convert results to our format
            detected_regions = []
            h, w = image_gray.shape
            
            for i, (bbox_coords, text, confidence) in enumerate(results):
                # EasyOCR returns bbox as [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                # Convert to [x, y, width, height] in percentage
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Convert to percentage coordinates
                x_percent = (x_min / w) * 100
                y_percent = (y_min / h) * 100
                width_percent = ((x_max - x_min) / w) * 100
                height_percent = ((y_max - y_min) / h) * 100
                
                detected_regions.append({
                    'text': text,
                    'bbox': [x_percent, y_percent, width_percent, height_percent],
                    'confidence': confidence,
                    'region_id': i + 1
                })
                
                if self.debug:
                    bbox_str = f"[{x_percent:.2f}, {y_percent:.2f}, {width_percent:.2f}, {height_percent:.2f}]"
                    print(f"   Debug - Region {i+1}: '{text}' at {bbox_str} (conf: {confidence:.3f})")
            
            # Create the map structure
            image_filename = os.path.basename(image_path)
            ocr_detected_map = {
                image_filename: detected_regions
            }
            
            return ocr_detected_map
            
        except Exception as e:
            if self.debug:
                print(f"   Debug - Error creating OCR detected map: {e}")
            return {}

    def save_ocr_detected_map(self, ocr_detected_map: Dict[str, list], output_dir: str) -> str:
        """
        Save the OCR detected map to a JSON file.
        For batch processing, this method accumulates results from multiple images.
        
        Args:
            ocr_detected_map: Dictionary containing detected text regions
            output_dir: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(output_dir, 'OCR_detected_map.json')
        
        try:
            # For batch processing, we want to accumulate results from multiple images
            # Load existing map if it exists
            existing_map = {}
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_map = json.load(f)
            
            # Merge new results with existing results
            merged_map = {**existing_map, **ocr_detected_map}
            
            # Save the merged map
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_map, f, indent=2, ensure_ascii=False)
            
            if self.debug:
                print(f"   Debug - OCR detected map saved to: {output_path}")
                print(f"   Debug - Total images in map: {len(merged_map)}")
            
            return output_path
            
        except Exception as e:
            if self.debug:
                print(f"   Debug - Error saving OCR detected map: {e}")
            return None

    def _check_overlap_text_aware(self, bbox1_px, bbox2_px, text1, text2, tolerance, 
                                text_similarity, iou, center_distance, size_similarity):
        """Text-aware matching strategy - prioritizes text similarity."""
        # If text similarity is very low, reject the match regardless of position
        if text_similarity < 0.05:  # 5% text similarity threshold
            return False, 0.0, "none", text_similarity
        
        # If IoU is significant, consider it a match (with text similarity bonus)
        if iou > 0.1:  # 10% overlap threshold
            combined_score = (iou * 0.6 + text_similarity * 0.4)
            return True, combined_score, "iou", text_similarity
        
        # Check for proximity with tolerance
        x1, y1, w1, h1 = bbox1_px
        x2, y2, w2, h2 = bbox2_px
        
        # Expand bounding boxes by tolerance
        x1_min, x1_max = x1 - tolerance, x1 + w1 + tolerance
        y1_min, y1_max = y1 - tolerance, y1 + h1 + tolerance
        x2_min, x2_max = x2 - tolerance, x2 + w2 + tolerance
        y2_min, y2_max = y2 - tolerance, y2 + h2 + tolerance
        
        # Check for overlap with tolerance
        tolerance_overlap = not (x1_max < x2_min or x2_max < x1_min or 
                               y1_max < y2_min or y2_max < y1_min)
        
        if tolerance_overlap:
            # Calculate a combined score for tolerance-based matches
            max_center_distance = max(w1, h1, w2, h2) * 0.5
            center_score = max(0, 1 - (center_distance / max_center_distance))
            position_score = (center_score + size_similarity) / 2
            combined_score = (position_score * 0.4 + text_similarity * 0.6)  # Prioritize text similarity
            
            if combined_score > 0.3:  # 30% combined similarity threshold
                return True, combined_score, "tolerance", text_similarity
        
        # Check for center-based matching
        if center_distance <= tolerance * 10:
            if size_similarity > 0.05:
                center_score = max(0, 1 - (center_distance / (tolerance * 10)))
                position_score = (center_score + size_similarity) / 2
                combined_score = (position_score * 0.3 + text_similarity * 0.7)  # Heavily prioritize text similarity
                
                if combined_score > 0.4:
                    return True, combined_score, "center", text_similarity
        
        # Check for position-based matching
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        
        x_distance = abs(center1_x - center2_x)
        y_distance = abs(center1_y - center2_y)
        
        if x_distance <= tolerance * 2 and y_distance <= tolerance * 2:
            if size_similarity > 0.05:
                position_score = max(0, 1 - (max(x_distance, y_distance) / (tolerance * 2)))
                pos_size_score = (position_score + size_similarity) / 2
                combined_score = (pos_size_score * 0.2 + text_similarity * 0.8)  # Very heavily prioritize text similarity
                
                if combined_score > 0.5:
                    return True, combined_score, "position", text_similarity
        
        return False, 0.0, "none", text_similarity

    def _check_overlap_position_focused(self, bbox1_px, bbox2_px, text1, text2, tolerance,
                                      text_similarity, iou, center_distance, size_similarity):
        """Position-focused matching strategy - prioritizes bounding box similarity."""
        # If IoU is significant, consider it a match (with text similarity bonus)
        if iou > 0.1:  # 10% overlap threshold
            combined_score = (iou * 0.8 + text_similarity * 0.2)  # Prioritize position
            return True, combined_score, "iou", text_similarity
        
        # Check for proximity with tolerance
        x1, y1, w1, h1 = bbox1_px
        x2, y2, w2, h2 = bbox2_px
        
        # Expand bounding boxes by tolerance
        x1_min, x1_max = x1 - tolerance, x1 + w1 + tolerance
        y1_min, y1_max = y1 - tolerance, y1 + h1 + tolerance
        x2_min, x2_max = x2 - tolerance, x2 + w2 + tolerance
        y2_min, y2_max = y2 - tolerance, y2 + h2 + tolerance
        
        # Check for overlap with tolerance
        tolerance_overlap = not (x1_max < x2_min or x2_max < x1_min or 
                               y1_max < y2_min or y2_max < y1_min)
        
        if tolerance_overlap:
            # Calculate a combined score for tolerance-based matches
            max_center_distance = max(w1, h1, w2, h2) * 0.5
            center_score = max(0, 1 - (center_distance / max_center_distance))
            position_score = (center_score + size_similarity) / 2
            combined_score = (position_score * 0.7 + text_similarity * 0.3)  # Prioritize position
            
            if combined_score > 0.3:  # 30% combined similarity threshold
                return True, combined_score, "tolerance", text_similarity
        
        # Check for center-based matching
        if center_distance <= tolerance * 10:
            if size_similarity > 0.05:
                center_score = max(0, 1 - (center_distance / (tolerance * 10)))
                position_score = (center_score + size_similarity) / 2
                combined_score = (position_score * 0.8 + text_similarity * 0.2)  # Heavily prioritize position
                
                if combined_score > 0.4:
                    return True, combined_score, "center", text_similarity
        
        # Check for position-based matching
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        
        x_distance = abs(center1_x - center2_x)
        y_distance = abs(center1_y - center2_y)
        
        if x_distance <= tolerance * 2 and y_distance <= tolerance * 2:
            if size_similarity > 0.05:
                position_score = max(0, 1 - (max(x_distance, y_distance) / (tolerance * 2)))
                pos_size_score = (position_score + size_similarity) / 2
                combined_score = (pos_size_score * 0.9 + text_similarity * 0.1)  # Very heavily prioritize position
                
                if combined_score > 0.5:
                    return True, combined_score, "position", text_similarity
        
        return False, 0.0, "none", text_similarity

    def compare_ground_truth_with_ocr(self, ground_truth_map: Dict[str, list], ocr_detected_map: Dict[str, list], image_path: str, tolerance: int = 1) -> Dict[str, any]:
        """
        Compare ground truth map with OCR detected map, focusing on overlapping bounding boxes.
        
        Args:
            ground_truth_map: Dictionary containing ground truth data
            ocr_detected_map: Dictionary containing OCR detected data
            image_path: Path to the original image for pixel calculations
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # Load image to get dimensions for pixel calculations
            image_rgb, image_gray = self.load_image(image_path)
            h, w = image_gray.shape
            
            image_filename = os.path.basename(image_path)
            
            # Get ground truth and OCR data for this image
            gt_entries = ground_truth_map.get(image_filename, [])
            ocr_entries = ocr_detected_map.get(image_filename, [])
            
            if self.debug:
                print(f"   Debug - Comparing {len(gt_entries)} ground truth entries with {len(ocr_entries)} OCR entries")
            
            comparison_results = {
                'image_filename': image_filename,
                'total_gt_regions': len(gt_entries),
                'total_ocr_regions': len(ocr_entries),
                'matching_regions': 0,
                'gt_only_regions': 0,
                'ocr_only_regions': 0,
                'overlap_threshold_pixels': tolerance,
                'matches': [],
                'gt_only': [],
                'ocr_only': []
            }
            
            # Convert percentage coordinates to pixel coordinates for comparison
            def percent_to_pixels(bbox_percent):
                x_pct, y_pct, w_pct, h_pct = bbox_percent
                x_px = int(x_pct * w / 100)
                y_px = int(y_pct * h / 100)
                w_px = int(w_pct * w / 100)
                h_px = int(h_pct * h / 100)
                return [x_px, y_px, w_px, h_px]
            
            def calculate_iou(bbox1_px, bbox2_px):
                """Calculate Intersection over Union (IoU) between two bounding boxes."""
                x1, y1, w1, h1 = bbox1_px
                x2, y2, w2, h2 = bbox2_px
                
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right < x_left or y_bottom < y_top:
                    return 0.0
                
                intersection = (x_right - x_left) * (y_bottom - y_top)
                
                # Calculate union
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            
            def calculate_center_distance(bbox1_px, bbox2_px):
                """Calculate the distance between the centers of two bounding boxes."""
                x1, y1, w1, h1 = bbox1_px
                x2, y2, w2, h2 = bbox2_px
                
                # Calculate centers
                center1_x = x1 + w1 / 2
                center1_y = y1 + h1 / 2
                center2_x = x2 + w2 / 2
                center2_y = y2 + h2 / 2
                
                # Calculate Euclidean distance
                distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
                return distance
            
            def calculate_size_similarity(bbox1_px, bbox2_px):
                """Calculate how similar two bounding boxes are in size."""
                x1, y1, w1, h1 = bbox1_px
                x2, y2, w2, h2 = bbox2_px
                
                # Calculate areas
                area1 = w1 * h1
                area2 = w2 * h2
                
                # Calculate size similarity (smaller area / larger area)
                if area1 == 0 or area2 == 0:
                    return 0.0
                
                min_area = min(area1, area2)
                max_area = max(area1, area2)
                return min_area / max_area
            
            def calculate_text_similarity(text1, text2):
                """Calculate text similarity between two strings."""
                if not text1 or not text2:
                    return 0.0
                
                # Convert to lowercase for comparison
                text1_lower = text1.lower().strip()
                text2_lower = text2.lower().strip()
                
                # If texts are identical, return 1.0
                if text1_lower == text2_lower:
                    return 1.0
                
                # Calculate Levenshtein distance
                max_len = max(len(text1_lower), len(text2_lower))
                if max_len == 0:
                    return 0.0
                
                char_distance = self._levenshtein_distance(text1_lower, text2_lower)
                char_similarity = 1.0 - (char_distance / max_len)
                
                # Calculate word-level similarity
                words1 = set(text1_lower.split())
                words2 = set(text2_lower.split())
                
                if not words1 or not words2:
                    word_similarity = 0.0
                else:
                    intersection = words1.intersection(words2)
                    union = words1.union(words2)
                    word_similarity = len(intersection) / len(union) if union else 0.0
                
                # Calculate substring similarity (for partial matches)
                substring_similarity = 0.0
                if len(text1_lower) > 2 and len(text2_lower) > 2:
                    # Check if one text contains the other as a substring
                    if text1_lower in text2_lower or text2_lower in text1_lower:
                        substring_similarity = min(len(text1_lower), len(text2_lower)) / max(len(text1_lower), len(text2_lower))
                    else:
                        # Check for common substrings
                        common_chars = 0
                        for i in range(min(len(text1_lower), len(text2_lower))):
                            if text1_lower[i] == text2_lower[i]:
                                common_chars += 1
                        if common_chars > 0:
                            substring_similarity = common_chars / max(len(text1_lower), len(text2_lower))
                        
                        # Check for partial word matches (e.g., "82ND3-319" vs "D 0 0 4 9" where "319" might match "4 9")
                        words1 = text1_lower.split()
                        words2 = text2_lower.split()
                        
                        # Look for any word that appears in both texts
                        for word1 in words1:
                            for word2 in words2:
                                if word1 in word2 or word2 in word1:
                                    substring_similarity = max(substring_similarity, min(len(word1), len(word2)) / max(len(word1), len(word2)))
                                    break
                
                # Combine similarities with weights
                combined_similarity = (char_similarity * 0.4 + word_similarity * 0.4 + substring_similarity * 0.2)
                return combined_similarity
            
            def check_overlap(bbox1_px, bbox2_px, text1, text2, tolerance=1):
                """Check if two bounding boxes overlap using multiple criteria with configurable strategy."""
                # Calculate text similarity first
                text_similarity = calculate_text_similarity(text1, text2)
                
                # Calculate IoU
                iou = calculate_iou(bbox1_px, bbox2_px)
                
                # Calculate center distance
                center_distance = calculate_center_distance(bbox1_px, bbox2_px)
                
                # Calculate size similarity
                size_similarity = calculate_size_similarity(bbox1_px, bbox2_px)
                
                # Choose matching strategy based on mode
                if self.matching_mode == "text_aware":
                    return self._check_overlap_text_aware(bbox1_px, bbox2_px, text1, text2, tolerance, 
                                                        text_similarity, iou, center_distance, size_similarity)
                else:  # position_focused
                    return self._check_overlap_position_focused(bbox1_px, bbox2_px, text1, text2, tolerance,
                                                              text_similarity, iou, center_distance, size_similarity)
            

            
            # Convert all bounding boxes to pixel coordinates
            gt_bboxes_px = []
            for gt_entry in gt_entries:
                if 'bbox' in gt_entry:
                    gt_bboxes_px.append(percent_to_pixels(gt_entry['bbox']))
                else:
                    gt_bboxes_px.append(None)
            
            ocr_bboxes_px = []
            for ocr_entry in ocr_entries:
                if 'bbox' in ocr_entry:
                    ocr_bboxes_px.append(percent_to_pixels(ocr_entry['bbox']))
                else:
                    ocr_bboxes_px.append(None)
            
            # Find matches based on overlapping bounding boxes
            matched_gt_indices = set()
            matched_ocr_indices = set()
            
            # First pass: find all potential matches for each GT region
            potential_matches = []
            
            for gt_idx, gt_bbox_px in enumerate(gt_bboxes_px):
                if gt_bbox_px is None:
                    continue
                
                gt_entry = gt_entries[gt_idx]
                gt_text = gt_entry.get('text', '')
                    
                for ocr_idx, ocr_bbox_px in enumerate(ocr_bboxes_px):
                    if ocr_bbox_px is None:
                        continue
                    
                    ocr_entry = ocr_entries[ocr_idx]
                    ocr_text = ocr_entry.get('text', '')
                    
                    is_match, match_score, match_type, text_similarity = check_overlap(gt_bbox_px, ocr_bbox_px, gt_text, ocr_text, tolerance=tolerance)
                    
                    if self.debug and not is_match:
                        # Show potential matches that were rejected
                        center_distance = calculate_center_distance(gt_bbox_px, ocr_bbox_px)
                        size_similarity = calculate_size_similarity(gt_bbox_px, ocr_bbox_px)
                        iou = calculate_iou(gt_bbox_px, ocr_bbox_px)
                        
                        # Only show if there's some similarity
                        if iou > 0.01 or center_distance < tolerance * 10 or size_similarity > 0.05 or text_similarity > 0.05:
                            print(f"   Debug - Potential match rejected: GT '{gt_text}' vs OCR '{ocr_text}' (IoU: {iou:.3f}, center_dist: {center_distance:.1f}, size_sim: {size_similarity:.3f}, text_sim: {text_similarity:.3f})")
                    
                    if is_match:
                        # Calculate IoU for this potential match
                        iou = calculate_iou(gt_bbox_px, ocr_bbox_px)
                        
                        # Calculate additional metrics
                        center_distance = calculate_center_distance(gt_bbox_px, ocr_bbox_px)
                        size_similarity = calculate_size_similarity(gt_bbox_px, ocr_bbox_px)
                        
                        potential_matches.append({
                            'gt_idx': gt_idx,
                            'ocr_idx': ocr_idx,
                            'iou': iou,
                            'match_score': match_score,
                            'match_type': match_type,
                            'center_distance': center_distance,
                            'size_similarity': size_similarity,
                            'text_similarity': text_similarity
                        })
            
            # Sort potential matches by combined match score (highest first) to prioritize better matches
            potential_matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Second pass: assign matches greedily, taking the best available match for each region
            for match in potential_matches:
                gt_idx = match['gt_idx']
                ocr_idx = match['ocr_idx']
                
                # Skip if either region is already matched
                if gt_idx in matched_gt_indices or ocr_idx in matched_ocr_indices:
                    continue
                
                # Found a match
                matched_gt_indices.add(gt_idx)
                matched_ocr_indices.add(ocr_idx)
                
                gt_entry = gt_entries[gt_idx]
                ocr_entry = ocr_entries[ocr_idx]
                
                # Get text from both ground truth and OCR entries
                gt_text = gt_entry.get('text', '')
                ocr_text = ocr_entry.get('text', '')
                ocr_confidence = ocr_entry.get('confidence', 0.0)
                
                # Calculate Levenshtein distance
                char_distance = self._levenshtein_distance(gt_text.lower(), ocr_text.lower())
                max_len = max(len(gt_text), len(ocr_text))
                normalized_distance = char_distance / max_len if max_len > 0 else 1.0
                
                # Word-level comparison
                gt_words = gt_text.lower().split()
                ocr_words = ocr_text.lower().split()
                
                word_matches = 0
                for gt_word in gt_words:
                    for ocr_word in ocr_words:
                        if gt_word == ocr_word:
                            word_matches += 1
                            break
                
                word_accuracy = word_matches / len(gt_words) if gt_words else 0.0
                
                # Get match metrics
                iou = match['iou']
                match_score = match['match_score']
                match_type = match['match_type']
                center_distance = match['center_distance']
                size_similarity = match['size_similarity']
                text_similarity = match.get('text_similarity', 0.0)
                
                match_result = {
                    'gt_index': gt_idx,
                    'ocr_index': ocr_idx,
                    'gt_text': gt_text,
                    'ocr_text': ocr_text,
                    'gt_bbox': gt_entry['bbox'],
                    'ocr_bbox': ocr_entry['bbox'],
                    'gt_bbox_px': gt_bboxes_px[gt_idx],
                    'ocr_bbox_px': ocr_bboxes_px[ocr_idx],
                    'iou': iou,
                    'match_score': match_score,
                    'match_type': match_type,
                    'center_distance': center_distance,
                    'size_similarity': size_similarity,
                    'text_similarity': text_similarity,
                    'ocr_confidence': ocr_confidence,
                    'char_distance': char_distance,
                    'normalized_distance': normalized_distance,
                    'word_accuracy': word_accuracy,
                    'word_matches': word_matches,
                    'total_gt_words': len(gt_words)
                }
                
                comparison_results['matches'].append(match_result)
                
                if self.debug:
                    print(f"   Debug - Match found: GT '{gt_text}' vs OCR '{ocr_text}' (conf: {ocr_confidence:.3f}, char_dist: {char_distance}, word_acc: {word_accuracy:.3f}, IoU: {iou:.3f}, match_score: {match_score:.3f}, type: {match_type}, center_dist: {center_distance:.1f}, size_sim: {size_similarity:.3f}, text_sim: {text_similarity:.3f})")
            
            # Find GT-only regions (not matched)
            for gt_idx, gt_entry in enumerate(gt_entries):
                if gt_idx not in matched_gt_indices:
                    comparison_results['gt_only'].append({
                        'index': gt_idx,
                        'text': gt_entry.get('text', ''),
                        'bbox': gt_entry.get('bbox', [])
                    })
            
            # Find OCR-only regions (not matched)
            for ocr_idx, ocr_entry in enumerate(ocr_entries):
                if ocr_idx not in matched_ocr_indices:
                    comparison_results['ocr_only'].append({
                        'index': ocr_idx,
                        'text': ocr_entry.get('text', ''),
                        'bbox': ocr_entry.get('bbox', []),
                        'confidence': ocr_entry.get('confidence', 0.0)
                    })
            
            # Update summary statistics
            comparison_results['matching_regions'] = len(comparison_results['matches'])
            comparison_results['gt_only_regions'] = len(comparison_results['gt_only'])
            comparison_results['ocr_only_regions'] = len(comparison_results['ocr_only'])
            
            # Calculate overall metrics
            if comparison_results['matches']:
                avg_char_distance = sum(m['char_distance'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_normalized_distance = sum(m['normalized_distance'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_word_accuracy = sum(m['word_accuracy'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_confidence = sum(m['ocr_confidence'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_iou = sum(m['iou'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_match_score = sum(m['match_score'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_center_distance = sum(m['center_distance'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_size_similarity = sum(m['size_similarity'] for m in comparison_results['matches']) / len(comparison_results['matches'])
                avg_text_similarity = sum(m.get('text_similarity', 0.0) for m in comparison_results['matches']) / len(comparison_results['matches'])
                
                # Count match types
                match_types = {}
                for m in comparison_results['matches']:
                    match_type = m['match_type']
                    match_types[match_type] = match_types.get(match_type, 0) + 1
                
                comparison_results['overall_metrics'] = {
                    'average_char_distance': avg_char_distance,
                    'average_normalized_distance': avg_normalized_distance,
                    'average_word_accuracy': avg_word_accuracy,
                    'average_ocr_confidence': avg_confidence,
                    'average_iou': avg_iou,
                    'average_match_score': avg_match_score,
                    'average_center_distance': avg_center_distance,
                    'average_size_similarity': avg_size_similarity,
                    'average_text_similarity': avg_text_similarity,
                    'match_types': match_types,
                    'match_rate': comparison_results['matching_regions'] / comparison_results['total_gt_regions'] if comparison_results['total_gt_regions'] > 0 else 0.0
                }
            else:
                comparison_results['overall_metrics'] = {
                    'average_char_distance': 0.0,
                    'average_normalized_distance': 1.0,
                    'average_word_accuracy': 0.0,
                    'average_ocr_confidence': 0.0,
                    'average_iou': 0.0,
                    'average_match_score': 0.0,
                    'average_center_distance': 0.0,
                    'average_size_similarity': 0.0,
                    'average_text_similarity': 0.0,
                    'match_types': {},
                    'match_rate': 0.0
                }
            
            if self.debug:
                print(f"   Debug - Comparison complete:")
                print(f"     Matches: {comparison_results['matching_regions']}")
                print(f"     GT only: {comparison_results['gt_only_regions']}")
                print(f"     OCR only: {comparison_results['ocr_only_regions']}")
                if comparison_results['overall_metrics']:
                    print(f"     Avg word accuracy: {comparison_results['overall_metrics']['average_word_accuracy']:.3f}")
                    print(f"     Avg confidence: {comparison_results['overall_metrics']['average_ocr_confidence']:.3f}")
                    print(f"     Avg IoU: {comparison_results['overall_metrics']['average_iou']:.3f}")
                    print(f"     Avg match score: {comparison_results['overall_metrics']['average_match_score']:.3f}")
                    print(f"     Avg center distance: {comparison_results['overall_metrics']['average_center_distance']:.1f}")
                    print(f"     Avg size similarity: {comparison_results['overall_metrics']['average_size_similarity']:.3f}")
                    print(f"     Avg text similarity: {comparison_results['overall_metrics']['average_text_similarity']:.3f}")
                    print(f"     Match types: {comparison_results['overall_metrics']['match_types']}")
            
            return comparison_results
            
        except Exception as e:
            if self.debug:
                print(f"   Debug - Error in ground truth comparison: {e}")
            return {
                'image_filename': os.path.basename(image_path),
                'total_gt_regions': 0,
                'total_ocr_regions': 0,
                'matching_regions': 0,
                'gt_only_regions': 0,
                'ocr_only_regions': 0,
                'matches': [],
                'gt_only': [],
                'ocr_only': [],
                'overall_metrics': {
                    'average_char_distance': 0.0,
                    'average_normalized_distance': 1.0,
                    'average_word_accuracy': 0.0,
                    'average_ocr_confidence': 0.0,
                    'match_rate': 0.0
                }
            }
            
        except Exception as e:
            if self.debug:
                print(f"   Debug - Error comparing ground truth with OCR: {e}")
            return {}

    def calculate_comprehensive_metrics(self, ground_truth_map: Dict[str, list], ocr_detected_map: Dict[str, list], image_path: str, comparison_results: dict = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics based on ground truth and OCR detected maps.
        Only consider detected (matched) and missed (gt_only) regions for region-level metrics.
        Ignore OCR-only regions for precision/recall/F1.
        """
        try:
            image_filename = os.path.basename(image_path)
            gt_entries = ground_truth_map.get(image_filename, [])
            ocr_entries = ocr_detected_map.get(image_filename, [])

            if self.debug:
                print(f"   Debug - Calculating comprehensive metrics for {len(gt_entries)} GT and {len(ocr_entries)} OCR entries")

            # Basic counts
            total_gt_regions = len(gt_entries)
            total_ocr_regions = len(ocr_entries)

            # Extract all text for global analysis
            gt_texts = [entry.get('text', '') for entry in gt_entries]
            ocr_texts = [entry.get('text', '') for entry in ocr_entries]
            ocr_confidences = [entry.get('confidence', 0.0) for entry in ocr_entries]

            # Combine all text
            combined_gt_text = ' '.join(gt_texts)
            combined_ocr_text = ' '.join(ocr_texts)
            
            # Debug text combination in comprehensive metrics
            if self.debug:
                print(f"   Debug - Comprehensive metrics text combination:")
                print(f"     GT texts from entries: {gt_texts}")
                print(f"     Combined GT text: '{combined_gt_text}'")
                print(f"     OCR texts from entries: {ocr_texts}")
                print(f"     Combined OCR text: '{combined_ocr_text}'")
                if hasattr(self, 'ground_truth_text_used'):
                    print(f"     Simple calculation GT text: '{self.ground_truth_text_used}'")
                    print(f"     Texts match: {combined_gt_text == self.ground_truth_text_used}")
                if hasattr(self, 'detected_text_used'):
                    print(f"     Simple calculation OCR text: '{self.detected_text_used}'")
                    print(f"     OCR texts match: {combined_ocr_text == self.detected_text_used}")

            # Character-level metrics (unchanged)
            char_distance = self._levenshtein_distance(combined_gt_text.lower(), combined_ocr_text.lower())
            max_char_len = max(len(combined_gt_text), len(combined_ocr_text))
            normalized_char_distance = char_distance / max_char_len if max_char_len > 0 else 1.0
            char_error_rate = char_distance / len(combined_gt_text) if len(combined_gt_text) > 0 else 1.0
            if np.isnan(normalized_char_distance):
                normalized_char_distance = 1.0
            if np.isnan(char_error_rate):
                char_error_rate = 1.0

            # Word-level metrics (UPDATED to use Levenshtein-based calculation)
            # CRITICAL FIX: Use the same text combination logic as the simple approach for consistency
            # The simple approach uses ground_truth_text which is already properly combined
            # So we should use the same combined text here instead of re-combining from individual entries
            
            # Get the ground truth text that was used in the simple approach
            # This ensures consistency between simple and comprehensive metrics
            if hasattr(self, 'ground_truth_text_used'):
                # Use the same ground truth text that was used in the simple calculation
                simple_gt_text = self.ground_truth_text_used
            else:
                # Fallback to re-combining, but this should be avoided
                simple_gt_text = combined_gt_text
            
            # CRITICAL FIX: Also use the same OCR text that was used in the simple calculation
            # The simple calculation uses detected_texts from OCR results, not from OCR map entries
            # We need to ensure we're using the same OCR text for consistency
            if hasattr(self, 'detected_text_used'):
                # Use the same detected text that was used in the simple calculation
                simple_ocr_text = self.detected_text_used
            else:
                # Fallback to re-combining from OCR map entries
                simple_ocr_text = combined_ocr_text
            
            word_metrics = self._calculate_word_level_metrics(simple_ocr_text, simple_gt_text)
            word_accuracy = word_metrics['word_accuracy']
            word_precision = word_metrics['word_precision']
            word_recall = word_metrics['word_recall']
            word_f1 = word_metrics['word_f1']
            # Calculate actual word matches based on accuracy and total GT words
            total_gt_words = word_metrics['ground_truth_word_count']
            total_ocr_words = word_metrics['detected_word_count']
            word_matches = int(word_accuracy * total_gt_words)  # Convert accuracy back to actual matches
            
            # Debug word accuracy calculation
            if self.debug:
                print(f"   Debug - Word accuracy calculation:")
                print(f"     Combined GT text: '{combined_gt_text}'")
                print(f"     Combined OCR text: '{combined_ocr_text}'")
                if hasattr(self, 'detected_text_used'):
                    print(f"     Simple calculation OCR text: '{self.detected_text_used}'")
                if hasattr(self, 'ground_truth_text_used'):
                    print(f"     Simple calculation GT text: '{self.ground_truth_text_used}'")
                print(f"     Word accuracy from _calculate_word_level_metrics: {word_accuracy}")
                print(f"     Word precision: {word_precision}")
                print(f"     Word recall: {word_recall}")
                print(f"     Word F1: {word_f1}")
                print(f"     Total GT words: {total_gt_words}")
                print(f"     Total OCR words: {total_ocr_words}")
                print(f"     Word matches: {word_matches}")
            
            # Keep the old distance calculations for backward compatibility
            gt_words = combined_gt_text.lower().split()
            ocr_words = combined_ocr_text.lower().split()
            word_distance = self._levenshtein_distance(' '.join(gt_words), ' '.join(ocr_words))
            max_word_len = max(len(gt_words), len(ocr_words))
            normalized_word_distance = word_distance / max_word_len if max_word_len > 0 else 1.0
            word_error_rate = word_distance / len(gt_words) if len(gt_words) > 0 else 1.0
            if np.isnan(normalized_word_distance):
                normalized_word_distance = 1.0
            if np.isnan(word_error_rate):
                word_error_rate = 1.0

            # --- REGION-LEVEL METRICS (UPDATED) ---
            # Use only matches and gt_only (ignore ocr_only)
            if comparison_results is not None:
                num_matches = len(comparison_results.get('matches', []))
                num_gt_only = len(comparison_results.get('gt_only', []))
                total_gt = num_matches + num_gt_only
                # Detection Rate = matches / total GT
                region_detection_rate = num_matches / total_gt if total_gt > 0 else 0.0
                # Precision = matches / (matches + gt_only) = matches / total GT
                region_precision = num_matches / total_gt if total_gt > 0 else 0.0
                # Recall = matches / total GT (same as detection rate)
                region_recall = region_detection_rate
                # F1
                region_f1 = 2 * (region_precision * region_recall) / (region_precision + region_recall) if (region_precision + region_recall) > 0 else 0.0
            else:
                # Fallback to old logic if no comparison_results provided
                region_detection_rate = total_ocr_regions / total_gt_regions if total_gt_regions > 0 else 0.0
                region_precision = total_gt_regions / total_ocr_regions if total_ocr_regions > 0 else 0.0
                region_recall = region_detection_rate
                region_f1 = 2 * (region_precision * region_recall) / (region_precision + region_recall) if (region_precision + region_recall) > 0 else 0.0
            if np.isnan(region_detection_rate):
                region_detection_rate = 0.0
            if np.isnan(region_precision):
                region_precision = 0.0
            if np.isnan(region_recall):
                region_recall = 0.0
            if np.isnan(region_f1):
                region_f1 = 0.0

            # Confidence analysis (unchanged)
            avg_confidence = np.mean(ocr_confidences) if ocr_confidences else 0.0
            min_confidence = np.min(ocr_confidences) if ocr_confidences else 0.0
            max_confidence = np.max(ocr_confidences) if ocr_confidences else 0.0
            confidence_std = np.std(ocr_confidences) if len(ocr_confidences) > 1 else 0.0
            if np.isnan(avg_confidence):
                avg_confidence = 0.0
            if np.isnan(min_confidence):
                min_confidence = 0.0
            if np.isnan(max_confidence):
                max_confidence = 0.0
            if np.isnan(confidence_std):
                confidence_std = 0.0

            # Text length analysis (unchanged)
            gt_text_lengths = [len(text) for text in gt_texts]
            ocr_text_lengths = [len(text) for text in ocr_texts]
            avg_gt_length = np.mean(gt_text_lengths) if gt_text_lengths else 0.0
            avg_ocr_length = np.mean(ocr_text_lengths) if ocr_text_lengths else 0.0
            try:
                if len(gt_text_lengths) > 1 and len(ocr_text_lengths) > 1 and len(gt_text_lengths) == len(ocr_text_lengths):
                    length_correlation = np.corrcoef(gt_text_lengths, ocr_text_lengths)[0, 1]
                    if np.isnan(length_correlation):
                        length_correlation = 0.0
                else:
                    length_correlation = 0.0
            except Exception:
                length_correlation = 0.0

            # CRITICAL FIX: Quality score calculation (use the correct word accuracy from simple calculation)
            # The word accuracy from comprehensive metrics may be overwritten later, so we need to use
            # the word accuracy that will be preserved in the final metrics
            quality_factors = []
            char_accuracy = 1.0 - normalized_char_distance
            if np.isnan(char_accuracy):
                char_accuracy = 0.0
            quality_factors.append(char_accuracy)
            
            # CRITICAL FIX: Use the word accuracy from simple calculation that will be preserved
            # This ensures the quality score is calculated using the correct word accuracy
            if hasattr(self, 'ground_truth_text_used') and hasattr(self, 'detected_text_used'):
                # Calculate word accuracy using the same text as simple calculation
                simple_word_metrics = self._calculate_word_level_metrics(self.detected_text_used, self.ground_truth_text_used)
                quality_word_accuracy = simple_word_metrics['word_accuracy']
                quality_word_f1 = simple_word_metrics['word_f1']
            else:
                # Fallback to comprehensive calculation values
                quality_word_accuracy = word_accuracy
                quality_word_f1 = word_f1
            
            if np.isnan(quality_word_accuracy):
                quality_word_accuracy = 0.0
            quality_factors.append(quality_word_accuracy)
            quality_factors.append(quality_word_f1)
            confidence_factor = avg_confidence
            if np.isnan(confidence_factor):
                confidence_factor = 0.0
            quality_factors.append(confidence_factor)
            detection_factor = min(region_detection_rate, 1.0)
            if np.isnan(detection_factor):
                detection_factor = 0.0
            quality_factors.append(detection_factor)
            try:
                if self.debug:
                    print(f"   Debug - Quality score calculation:")
                    print(f"     Quality factors: {quality_factors}")
                    print(f"     All factors >= 0: {all(factor >= 0 for factor in quality_factors)}")
                
                if quality_factors and all(factor >= 0 for factor in quality_factors):
                    # Use arithmetic mean instead of geometric mean for more intuitive scoring
                    overall_quality = np.mean(quality_factors)
                    if np.isnan(overall_quality):
                        overall_quality = 0.0
                else:
                    # If any factor is negative, cap them at 0 and recalculate
                    if self.debug:
                        print(f"   Debug - Some quality factors are negative, capping at 0")
                    capped_factors = [max(0.0, factor) for factor in quality_factors]
                    overall_quality = np.mean(capped_factors) if capped_factors else 0.0
                    if np.isnan(overall_quality):
                        overall_quality = 0.0
            except Exception as e:
                if self.debug:
                    print(f"   Debug - Error in quality score calculation: {e}")
                overall_quality = 0.0

            comprehensive_metrics = {
                # Basic counts
                'total_gt_regions': total_gt_regions,
                'total_ocr_regions': total_ocr_regions,
                'gt_text_count': len(gt_texts),
                'ocr_text_count': len(ocr_texts),
                # Character-level metrics
                'character_distance': char_distance,
                'normalized_character_distance': normalized_char_distance,
                'character_error_rate': char_error_rate,
                'character_accuracy': char_accuracy,
                # Word-level metrics
                'word_distance': word_distance,
                'normalized_word_distance': normalized_word_distance,
                'word_error_rate': word_error_rate,
                'word_accuracy': word_accuracy,
                'word_precision': word_precision,
                'word_recall': word_recall,
                'word_f1': word_f1,
                'word_matches': word_matches,
                'total_gt_words': len(gt_words),
                'total_ocr_words': len(ocr_words),
                # Region-level metrics (updated)
                'region_detection_rate': region_detection_rate,
                'region_precision': region_precision,
                'region_recall': region_recall,
                'region_f1': region_f1,
                # Confidence metrics
                'average_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'confidence_std': confidence_std,
                # Text length analysis
                'average_gt_text_length': avg_gt_length,
                'average_ocr_text_length': avg_ocr_length,
                'text_length_correlation': length_correlation,
                # Quality scores
                'overall_quality_score': overall_quality,
                'quality_factors': quality_factors
            }

            if self.debug:
                print(f"   Debug - Comprehensive metrics calculated:")
                print(f"     Character accuracy: {char_accuracy:.3f}")
                print(f"     Word accuracy: {word_accuracy:.3f}")
                print(f"     Word precision: {word_precision:.3f} (Total OCR: {total_ocr_words})")
                print(f"     Word recall: {word_recall:.3f} (Total GT: {total_gt_words})")
                print(f"     Word F1: {word_f1:.3f}")
                print(f"     Word matches: {word_matches}/{total_gt_words}")
                print(f"     Avg confidence: {avg_confidence:.3f}")
                print(f"     Overall quality: {overall_quality:.3f}")

            return comprehensive_metrics
        
        except Exception as e:
            if self.debug:
                print(f"   Debug - Error calculating comprehensive metrics: {e}")
            return {
                'total_gt_regions': 0,
                'total_ocr_regions': 0,
                'overall_quality_score': 0.0
            }

    def save_comparison_results(self, comparison_results: Dict[str, any], output_dir: str) -> str:
        """
        Save the comparison results to a JSON file.
        
        Args:
            comparison_results: Dictionary containing comparison results
            output_dir: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        # Extract image filename from comparison results to create unique filename
        image_filename = comparison_results.get('image_filename', 'unknown_image')
        base_filename = os.path.splitext(image_filename)[0]  # Remove extension
        output_path = os.path.join(output_dir, f'ground_truth_ocr_comparison_{base_filename}.json')
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            
            if self.debug:
                print(f"   Debug - Comparison results saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            if self.debug:
                print(f"   Debug - Error saving comparison results: {e}")
            return None

    def evaluate_image(self, image_path: str, ground_truth_text: str = None, ground_truth_bboxes: list = None, output_dir: str = None, ground_truth_entries: list = None, tolerance: int = 1) -> dict:
        print(f"Evaluating image: {image_path}")
        image_rgb, image_gray = self.load_image(image_path)
        
        # Set ground truth entries if provided
        if ground_truth_entries is not None:
            self.ground_truth_entries = ground_truth_entries
        
        # Step 1: Run OCR on entire image and create detected map
        print("   Step 1: Running OCR on entire image to detect text regions...")
        ocr_detected_map = self.create_ocr_detected_map(image_gray, image_path)
        
        # Save OCR detected map if output directory is provided
        if output_dir and ocr_detected_map:
            self.save_ocr_detected_map(ocr_detected_map, output_dir)
        
        # Step 2: Calculate image quality metrics
        print("   Step 2: Calculating image quality metrics...")
        self.metrics = {
            'laplacian_variance': self.calculate_laplacian_variance(image_gray),
            'contrast': self.calculate_contrast(image_gray),
            'noise_estimation': self.calculate_noise_estimation(image_gray),
            'structural_similarity_index': self.calculate_structural_similarity_index(image_gray),
            'multiscale_ssim': self.calculate_multiscale_ssim(image_gray)
        }
        additional_metrics = self.calculate_additional_metrics(image_gray)
        self.metrics.update(additional_metrics)
        
        # Step 3: Perform OCR analysis with ground truth comparison
        print("   Step 3: Performing OCR analysis with ground truth comparison...")
        # Store the ground truth text used in simple calculation for consistency
        self.ground_truth_text_used = ground_truth_text
        ocr_metrics = self.calculate_levenshtein_distance(image_gray, ground_truth_text, ground_truth_bboxes)
        # CRITICAL FIX: Store the detected text used in simple calculation for consistency
        self.detected_text_used = ocr_metrics.get('detected_text', '')
        self.metrics.update(ocr_metrics)
        
        # Step 4: Compare ground truth map with OCR detected map (if ground truth map is available)
        if hasattr(self, 'ground_truth_entries') and self.ground_truth_entries and ocr_detected_map:
            print("   Step 4: Comparing ground truth map with OCR detected map...")
            
            # Create ground truth map structure
            image_filename = os.path.basename(image_path)
            ground_truth_map = {image_filename: self.ground_truth_entries}
            
            # Perform detailed comparison
            comparison_results = self.compare_ground_truth_with_ocr(ground_truth_map, ocr_detected_map, image_path, tolerance)
            
            # Calculate comprehensive metrics based on both maps
            print("   Step 4a: Calculating comprehensive evaluation metrics...")
            comprehensive_metrics = self.calculate_comprehensive_metrics(ground_truth_map, ocr_detected_map, image_path, comparison_results)
            
            # Save comparison results
            if output_dir and comparison_results:
                self.save_comparison_results(comparison_results, output_dir)
                
                # Add both comparison and comprehensive metrics to main metrics
                if 'overall_metrics' in comparison_results:
                    self.metrics.update({
                        'gt_ocr_match_rate': comparison_results['overall_metrics']['match_rate'],
                        'gt_ocr_avg_word_accuracy': comparison_results['overall_metrics']['average_word_accuracy'],
                        'gt_ocr_avg_confidence': comparison_results['overall_metrics']['average_ocr_confidence'],
                        'gt_ocr_matching_regions': comparison_results['matching_regions'],
                        'gt_ocr_gt_only_regions': comparison_results['gt_only_regions'],
                        'gt_ocr_ocr_only_regions': comparison_results['ocr_only_regions']
                    })
                
                # CRITICAL FIX: Add comprehensive metrics, but preserve the correct word accuracy from simple calculation
                # Store the correct word accuracy before overwriting
                correct_word_accuracy = self.metrics.get('word_accuracy', 0.0)
                correct_word_precision = self.metrics.get('word_precision', 0.0)
                correct_word_recall = self.metrics.get('word_recall', 0.0)
                correct_word_f1 = self.metrics.get('word_f1', 0.0)
                
                # Update with comprehensive metrics
                self.metrics.update(comprehensive_metrics)
                
                # CRITICAL FIX: Restore the correct word-level metrics from simple calculation
                # This ensures consistency between simple and comprehensive approaches
                if self.debug:
                    print(f"   Debug - Word accuracy preservation:")
                    print(f"     Simple calculation word accuracy: {correct_word_accuracy}")
                    print(f"     Comprehensive calculation word accuracy: {comprehensive_metrics.get('word_accuracy', 'NOT FOUND')}")
                    print(f"     Restoring simple calculation values for consistency")
                
                self.metrics['word_accuracy'] = correct_word_accuracy
                self.metrics['word_precision'] = correct_word_precision
                self.metrics['word_recall'] = correct_word_recall
                self.metrics['word_f1'] = correct_word_f1
                
                # Create histogram visualization if output directory is provided
                if output_dir and ocr_detected_map:
                    print("   Step 4b: Creating histogram visualization...")
                    histogram_path = os.path.join(output_dir, f'histogram_analysis_{os.path.basename(image_path).split(".")[0]}.png')
                    self.create_histogram_visualization(ground_truth_map, ocr_detected_map, image_path, histogram_path)
                
                if self.debug:
                    print(f"   Debug - Added {len(comprehensive_metrics)} comprehensive metrics to evaluation")
        else:
            print("   Step 4: Skipping ground truth comparison (no ground truth map available)")

        # Debug: Print ground truth entries loaded for this image
        print("Loaded ground truth entries for this image:")
        print(self.ground_truth_entries if hasattr(self, 'ground_truth_entries') else None)

        # Always pair bboxes and texts from ground_truth_entries
        paired_bboxes = []
        paired_texts = []
        if hasattr(self, 'ground_truth_entries') and self.ground_truth_entries:
            for entry in self.ground_truth_entries:
                if 'bbox' in entry and 'text' in entry:
                    paired_bboxes.append(entry['bbox'])
                    paired_texts.append(entry['text'])
        else:
            paired_bboxes = ground_truth_bboxes if ground_truth_bboxes else []
            paired_texts = ['No GT'] * len(paired_bboxes)

        # Debug: Print the paired bboxes and texts
        print("Paired bounding boxes and texts for visualization:")
        for bbox, text in zip(paired_bboxes, paired_texts):
            print(f"BBox: {bbox} -> Text: {text}")

        # Check for bbox-text consistency
        bbox_text_map = {}
        if hasattr(self, 'ground_truth_entries') and self.ground_truth_entries:
            for entry in self.ground_truth_entries:
                bbox = tuple(entry['bbox']) if 'bbox' in entry else None
                text = entry.get('text', None)
                if bbox is None or text is None:
                    print(f"WARNING: Entry missing bbox or text: {entry}")
                    continue
                if bbox in bbox_text_map and bbox_text_map[bbox] != text:
                    print(f"WARNING: Duplicate bbox with different texts: {bbox} -> '{bbox_text_map[bbox]}' and '{text}'")
                bbox_text_map[bbox] = text
        if len(paired_bboxes) != len(paired_texts):
            print(f"WARNING: Number of bounding boxes ({len(paired_bboxes)}) and texts ({len(paired_texts)}) do not match!")

        if paired_bboxes:
            print("   Performing regional OCR analysis...")
            regional_metrics = self.analyze_regional_ocr(image_gray, paired_bboxes)
            if self.debug:
                print(f"   Debug - Regional metrics returned: {regional_metrics}")
            self.metrics.update(regional_metrics)
            # Get OCR results for each region (for display)
            ocr_results = []
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=self.use_cuda)
                regions = self.extract_bbox_regions(image_gray, paired_bboxes)
                for region in regions:
                    region_results = reader.readtext(region)
                    if region_results:
                        detected_texts = [text for _, text, _ in region_results]
                        ocr_results.append(" | ".join(detected_texts))
                    else:
                        ocr_results.append("No text detected")
            except Exception as e:
                if self.debug:
                    print(f"   Debug - Could not get OCR results for visualization: {e}")
                ocr_results = ["OCR error"] * len(paired_bboxes)
            if self.debug:
                bbox_viz_path = os.path.join(output_dir, 'bbox_visualization.png') if output_dir else os.path.join(os.path.dirname(image_path), 'bbox_visualization.png')
                self.visualize_bbox_regions(image_gray, paired_bboxes, paired_texts, ocr_results, bbox_viz_path)
        return self.metrics
    
    def clear_gpu_memory(self):
        """Clear GPU memory if using CUDA."""
        if self.use_cuda:
            torch.cuda.empty_cache()
    
    def print_results(self):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("OCR IMAGE QUALITY EVALUATION RESULTS")
        print("="*60)
        
        for metric, value in self.metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"{metric.replace('_', ' ').title():<30}: {value:.6f}")
            else:
                print(f"{metric.replace('_', ' ').title():<30}: {value}")
        
        # Print CER, WER, character recall, and F1 with descriptive labels if present
        if 'character_error_rate' in self.metrics:
            print(f"Character Error Rate (CER)           : {self.metrics['character_error_rate']:.6f}")
        if 'word_error_rate' in self.metrics:
            print(f"Word Error Rate (WER)                : {self.metrics['word_error_rate']:.6f}")
        if 'character_recall' in self.metrics:
            print(f"Character Recall                     : {self.metrics['character_recall']:.6f}")
        if 'word_f1' in self.metrics:
            print(f"Word F1 Score                        : {self.metrics['word_f1']:.6f}")
        
        # Print regional OCR metrics if available
        if 'total_regions' in self.metrics:
            print(f"Total Ground Truth Regions           : {self.metrics['total_regions']}")
        if 'regions_with_text' in self.metrics:
            print(f"Regions with Detected Text           : {self.metrics['regions_with_text']}")
        if 'region_detection_accuracy' in self.metrics:
            print(f"Region Detection Accuracy            : {self.metrics['region_detection_accuracy']:.6f}")
        if 'average_confidence_per_region' in self.metrics:
            print(f"Average Confidence per Region        : {self.metrics['average_confidence_per_region']:.6f}")
        
        # Print comprehensive map-based metrics
        print("\n" + "="*60)
        print("COMPREHENSIVE MAP-BASED EVALUATION")
        print("="*60)
        
        if 'total_gt_regions' in self.metrics:
            print(f"Ground Truth Regions                 : {self.metrics['total_gt_regions']}")
        if 'total_ocr_regions' in self.metrics:
            print(f"OCR Detected Regions                 : {self.metrics['total_ocr_regions']}")
        if 'gt_ocr_matching_regions' in self.metrics:
            print(f"Matching Regions                     : {self.metrics['gt_ocr_matching_regions']}")
        if 'gt_ocr_gt_only_regions' in self.metrics:
            print(f"GT Only Regions                      : {self.metrics['gt_ocr_gt_only_regions']}")
        if 'gt_ocr_ocr_only_regions' in self.metrics:
            print(f"OCR Only Regions                     : {self.metrics['gt_ocr_ocr_only_regions']}")
        
        # Character-level metrics
        if 'character_accuracy' in self.metrics:
            print(f"Character Accuracy                   : {self.metrics['character_accuracy']:.6f}")
        if 'character_error_rate' in self.metrics:
            print(f"Character Error Rate (CER)           : {self.metrics['character_error_rate']:.6f}")
        
        # Word-level metrics
        if 'word_accuracy' in self.metrics:
            print(f"Word Accuracy                        : {self.metrics['word_accuracy']:.6f}")
        if 'word_error_rate' in self.metrics:
            print(f"Word Error Rate (WER)                : {self.metrics['word_error_rate']:.6f}")
        if 'word_f1' in self.metrics:
            print(f"Word F1 Score                        : {self.metrics['word_f1']:.6f}")
        if 'word_precision' in self.metrics:
            print(f"Word Precision                       : {self.metrics['word_precision']:.6f}")
        if 'word_recall' in self.metrics:
            print(f"Word Recall                          : {self.metrics['word_recall']:.6f}")
        
        # Region-level metrics
        if 'region_detection_rate' in self.metrics:
            print(f"Region Detection Rate                : {self.metrics['region_detection_rate']:.6f}")
        if 'region_precision' in self.metrics:
            print(f"Region Precision                     : {self.metrics['region_precision']:.6f}")
        
        # Confidence metrics
        if 'average_confidence' in self.metrics:
            print(f"Average OCR Confidence               : {self.metrics['average_confidence']:.6f}")
        if 'min_confidence' in self.metrics:
            print(f"Minimum OCR Confidence               : {self.metrics['min_confidence']:.6f}")
        if 'max_confidence' in self.metrics:
            print(f"Maximum OCR Confidence               : {self.metrics['max_confidence']:.6f}")
        if 'confidence_std' in self.metrics:
            print(f"OCR Confidence Std Dev               : {self.metrics['confidence_std']:.6f}")
        
        # Quality scores
        if 'overall_quality_score' in self.metrics:
            print(f"Overall Quality Score                : {self.metrics['overall_quality_score']:.6f}")
        
        print("="*60)
        
        # Provide interpretation
        self._interpret_results()
    
    def _interpret_results(self):
        """Provide interpretation of the results."""
        print("\nINTERPRETATION:")
        print("-" * 40)
        
        # Laplacian Variance
        lv = self.metrics['laplacian_variance']
        if lv > 1000:
            print("✓ Laplacian Variance: High sharpness (good for OCR)")
        elif lv > 500:
            print("✓ Laplacian Variance: Moderate sharpness (acceptable for OCR)")
        else:
            print("✗ Laplacian Variance: Low sharpness (may affect OCR accuracy)")
        
        # Contrast
        contrast = self.metrics['contrast']
        if contrast > 50:
            print("✓ Contrast: High contrast (good for OCR)")
        elif contrast > 30:
            print("✓ Contrast: Moderate contrast (acceptable for OCR)")
        else:
            print("✗ Contrast: Low contrast (may affect OCR accuracy)")
        
        # Noise
        noise = self.metrics['noise_estimation']
        if noise < 5:
            print("✓ Noise: Low noise (good for OCR)")
        elif noise < 10:
            print("✓ Noise: Moderate noise (acceptable for OCR)")
        else:
            print("✗ Noise: High noise (may affect OCR accuracy)")
        
        # SSIM
        ssim_val = self.metrics['structural_similarity_index']
        if ssim_val > 0.8:
            print("✓ SSIM: High structural similarity (good for OCR)")
        elif ssim_val > 0.6:
            print("✓ SSIM: Moderate structural similarity (acceptable for OCR)")
        else:
            print("✗ SSIM: Low structural similarity (may affect OCR accuracy)")
        

        
        # OCR Confidence
        if 'ocr_confidence' in self.metrics:
            ocr_conf = self.metrics['ocr_confidence']
            if ocr_conf > 0.8:
                print("✓ OCR Confidence: High confidence (good for OCR)")
            elif ocr_conf > 0.6:
                print("✓ OCR Confidence: Moderate confidence (acceptable for OCR)")
            else:
                print("✗ OCR Confidence: Low confidence (may affect OCR accuracy)")
        
        # Text Detection Quality
        if 'text_detection_quality' in self.metrics:
            text_quality = self.metrics['text_detection_quality']
            if text_quality > 0.8:
                print("✓ Text Detection Quality: High quality (good for OCR)")
            elif text_quality > 0.6:
                print("✓ Text Detection Quality: Moderate quality (acceptable for OCR)")
            else:
                print("✗ Text Detection Quality: Low quality (may affect OCR accuracy)")
        
        # Regional Detection Accuracy
        if 'region_detection_accuracy' in self.metrics:
            region_acc = self.metrics['region_detection_accuracy']
            if region_acc > 0.8:
                print("✓ Regional Detection: High accuracy (good for OCR)")
            elif region_acc > 0.6:
                print("✓ Regional Detection: Moderate accuracy (acceptable for OCR)")
            else:
                print("✗ Regional Detection: Low accuracy (may affect OCR accuracy)")
        
        # Average Confidence per Region
        if 'average_confidence_per_region' in self.metrics:
            avg_conf = self.metrics['average_confidence_per_region']
            if avg_conf > 0.8:
                print("✓ Regional Confidence: High confidence (good for OCR)")
            elif avg_conf > 0.6:
                print("✓ Regional Confidence: Moderate confidence (acceptable for OCR)")
            else:
                print("✗ Regional Confidence: Low confidence (may affect OCR accuracy)")
        
        # Comprehensive Map-Based Metrics
        print("\nMAP-BASED EVALUATION INTERPRETATION:")
        print("-" * 40)
        
        # Character Accuracy
        if 'character_accuracy' in self.metrics:
            char_acc = self.metrics['character_accuracy']
            if char_acc > 0.9:
                print("✓ Character Accuracy: Excellent (very good for OCR)")
            elif char_acc > 0.8:
                print("✓ Character Accuracy: Good (good for OCR)")
            elif char_acc > 0.7:
                print("⚠ Character Accuracy: Fair (acceptable for OCR)")
            else:
                print("✗ Character Accuracy: Poor (may affect OCR accuracy)")
        
        # Word F1 Score
        if 'word_f1' in self.metrics:
            word_f1 = self.metrics['word_f1']
            if word_f1 > 0.9:
                print("✓ Word F1 Score: Excellent word recognition")
            elif word_f1 > 0.8:
                print("✓ Word F1 Score: Good word recognition")
            elif word_f1 > 0.7:
                print("⚠ Word F1 Score: Fair word recognition")
            else:
                print("✗ Word F1 Score: Poor word recognition")
        
        # Region Detection Rate
        if 'region_detection_rate' in self.metrics:
            det_rate = self.metrics['region_detection_rate']
            if det_rate > 0.9:
                print("✓ Region Detection: Excellent (found most text regions)")
            elif det_rate > 0.8:
                print("✓ Region Detection: Good (found most text regions)")
            elif det_rate > 0.7:
                print("⚠ Region Detection: Fair (missed some text regions)")
            else:
                print("✗ Region Detection: Poor (missed many text regions)")
        
        # Overall Quality Score
        if 'overall_quality_score' in self.metrics:
            quality = self.metrics['overall_quality_score']
            if quality > 0.8:
                print("✓ Overall Quality: Excellent OCR performance")
            elif quality > 0.7:
                print("✓ Overall Quality: Good OCR performance")
            elif quality > 0.6:
                print("⚠ Overall Quality: Fair OCR performance")
            else:
                print("✗ Overall Quality: Poor OCR performance")
        
        # Confidence Analysis
        if 'average_confidence' in self.metrics:
            avg_conf = self.metrics['average_confidence']
            if avg_conf > 0.8:
                print("✓ OCR Confidence: High confidence in detections")
            elif avg_conf > 0.6:
                print("✓ OCR Confidence: Moderate confidence in detections")
            else:
                print("✗ OCR Confidence: Low confidence in detections")
    
    def save_results(self, output_path: str):
        """Save results to a JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def create_visualization(self, image_path: str, output_path: str = None):
        """Create a comprehensive visualization using map-based evaluation metrics."""
        
        # Create figure with subplots for different metric categories
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OCR Evaluation: Comprehensive Map-Based Analysis', fontsize=18, fontweight='bold', y=0.95)
        
        # Check if we have comprehensive map-based metrics
        has_map_metrics = 'overall_quality_score' in self.metrics
        has_region_metrics = 'total_gt_regions' in self.metrics
        
        # Debug: Print available metrics for visualization
        if self.debug:
            print(f"   Debug - Available metrics for visualization:")
            word_metrics = ['word_accuracy', 'word_precision', 'word_recall', 'word_f1']
            for metric in word_metrics:
                if metric in self.metrics:
                    print(f"     {metric}: {self.metrics[metric]}")
                else:
                    print(f"     {metric}: NOT FOUND")
        
        if has_map_metrics:
            # Plot 1: Character & Word Level Accuracy
            axes[0, 0].set_title('Character & Word Level Performance', fontweight='bold', pad=15)
            
            # Prepare data for accuracy metrics
            accuracy_metrics = []
            accuracy_values = []
            accuracy_colors = []
            
            if 'character_accuracy' in self.metrics:
                char_accuracy = self.metrics['character_accuracy']
                # Cap character accuracy between 0.0 and 1.0 for visualization
                char_accuracy = max(0.0, min(char_accuracy, 1.0))
                accuracy_metrics.append('Character\nAccuracy')
                accuracy_values.append(char_accuracy)
                accuracy_colors.append('#FF6B6B')  # Red
                if self.debug:
                    print(f"   Debug - Adding character accuracy to visualization: {char_accuracy:.3f} (capped from {self.metrics['character_accuracy']:.3f})")
            
            if 'word_accuracy' in self.metrics:
                word_acc_value = self.metrics['word_accuracy']
                # Cap word accuracy between 0.0 and 1.0 for visualization
                word_acc_value = max(0.0, min(word_acc_value, 1.0))
                # Always include word accuracy, even if it's 0
                accuracy_metrics.append('Word\nAccuracy')
                accuracy_values.append(word_acc_value)
                accuracy_colors.append('#4ECDC4')  # Teal
                if self.debug:
                    print(f"   Debug - Adding word accuracy to visualization: {word_acc_value:.3f} (capped from {self.metrics['word_accuracy']:.3f})")
            else:
                if self.debug:
                    print(f"   Debug - Word accuracy not found in metrics")
            
            if 'word_f1' in self.metrics:
                word_f1_value = self.metrics['word_f1']
                # Cap word F1 between 0.0 and 1.0 for visualization
                word_f1_value = max(0.0, min(word_f1_value, 1.0))
                accuracy_metrics.append('Word F1\nScore')
                accuracy_values.append(word_f1_value)
                accuracy_colors.append('#45B7D1')  # Blue
                if self.debug:
                    print(f"   Debug - Adding word F1 to visualization: {word_f1_value:.3f} (capped from {self.metrics['word_f1']:.3f})")
            
            if 'word_precision' in self.metrics:
                word_prec_value = self.metrics['word_precision']
                # Cap word precision between 0.0 and 1.0 for visualization
                word_prec_value = max(0.0, min(word_prec_value, 1.0))
                accuracy_metrics.append('Word\nPrecision')
                accuracy_values.append(word_prec_value)
                accuracy_colors.append('#96CEB4')  # Green
                if self.debug:
                    print(f"   Debug - Adding word precision to visualization: {word_prec_value:.3f} (capped from {self.metrics['word_precision']:.3f})")
            
            if 'word_recall' in self.metrics:
                word_recall_value = self.metrics['word_recall']
                # Cap word recall between 0.0 and 1.0 for visualization
                word_recall_value = max(0.0, min(word_recall_value, 1.0))
                accuracy_metrics.append('Word\nRecall')
                accuracy_values.append(word_recall_value)
                accuracy_colors.append('#FFEAA7')  # Yellow
                if self.debug:
                    print(f"   Debug - Adding word recall to visualization: {word_recall_value:.3f} (capped from {self.metrics['word_recall']:.3f})")
            
            if accuracy_metrics:
                bars = axes[0, 0].bar(accuracy_metrics, accuracy_values, color=accuracy_colors, alpha=0.8)
                axes[0, 0].set_ylabel('Accuracy Score')
                axes[0, 0].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, accuracy_values):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # Add threshold lines
                axes[0, 0].axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (0.9)')
                axes[0, 0].axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good (0.8)')
                axes[0, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Fair (0.7)')
                axes[0, 0].legend(fontsize=9)
            else:
                # No accuracy metrics found
                axes[0, 0].text(0.5, 0.5, 'No Word-Level Metrics\nAvailable', 
                               ha='center', va='center', transform=axes[0, 0].transAxes,
                               fontsize=12, fontweight='bold', color='red')
                if self.debug:
                    print(f"   Debug - No accuracy metrics found for visualization")
            
            # Plot 2: Region Detection & Confidence Analysis
            axes[0, 1].set_title('Region Detection & Confidence', fontweight='bold', pad=15)
            
            # Prepare data for region and confidence metrics
            region_metrics = []
            region_values = []
            region_colors = []
            
            if 'region_detection_rate' in self.metrics:
                region_detection_rate = self.metrics['region_detection_rate']
                # Cap region detection rate between 0.0 and 1.0 for visualization
                region_detection_rate = max(0.0, min(region_detection_rate, 1.0))
                region_metrics.append('Region\nDetection Rate')
                region_values.append(region_detection_rate)
                region_colors.append('#A259F7')  # Purple
                if self.debug:
                    print(f"   Debug - Adding region detection rate to visualization: {region_detection_rate:.3f} (capped from {self.metrics['region_detection_rate']:.3f})")
            
            if 'region_precision' in self.metrics:
                region_precision = self.metrics['region_precision']
                # Cap region precision between 0.0 and 1.0 for visualization
                region_precision = max(0.0, min(region_precision, 1.0))
                region_metrics.append('Region\nPrecision')
                region_values.append(region_precision)
                region_colors.append('#FFD700')  # Gold
                if self.debug:
                    print(f"   Debug - Adding region precision to visualization: {region_precision:.3f} (capped from {self.metrics['region_precision']:.3f})")
            
            if 'average_confidence' in self.metrics:
                avg_confidence = self.metrics['average_confidence']
                # Cap average confidence between 0.0 and 1.0 for visualization
                avg_confidence = max(0.0, min(avg_confidence, 1.0))
                region_metrics.append('Average\nConfidence')
                region_values.append(avg_confidence)
                region_colors.append('#DDA0DD')  # Plum
                if self.debug:
                    print(f"   Debug - Adding average confidence to visualization: {avg_confidence:.3f} (capped from {self.metrics['average_confidence']:.3f})")
            
            if 'min_confidence' in self.metrics:
                min_confidence = self.metrics['min_confidence']
                # Cap min confidence between 0.0 and 1.0 for visualization
                min_confidence = max(0.0, min(min_confidence, 1.0))
                region_metrics.append('Min\nConfidence')
                region_values.append(min_confidence)
                region_colors.append('#FFA500')  # Orange
                if self.debug:
                    print(f"   Debug - Adding min confidence to visualization: {min_confidence:.3f} (capped from {self.metrics['min_confidence']:.3f})")
            
            if region_metrics:
                bars = axes[0, 1].bar(region_metrics, region_values, color=region_colors, alpha=0.8)
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, region_values):
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # Add threshold lines
                axes[0, 1].axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent (0.9)')
                axes[0, 1].axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good (0.8)')
                axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Fair (0.7)')
                axes[0, 1].legend(fontsize=9)
            
            # Plot 3: Comprehensive Summary
            axes[1, 0].set_title('Comprehensive Evaluation Summary', fontweight='bold', pad=15)
            axes[1, 0].axis('off')
            
            summary_text = "MAP-BASED EVALUATION SUMMARY:\n\n"
            
            # Region counts
            if 'total_gt_regions' in self.metrics:
                summary_text += f"Ground Truth Regions: {self.metrics['total_gt_regions']}\n"
            if 'total_ocr_regions' in self.metrics:
                summary_text += f"OCR Detected Regions: {self.metrics['total_ocr_regions']}\n"
            if 'gt_ocr_matching_regions' in self.metrics:
                summary_text += f"Matching Regions: {self.metrics['gt_ocr_matching_regions']}\n"
            
            # Word counts
            if 'total_gt_words' in self.metrics:
                summary_text += f"Ground Truth Words: {self.metrics['total_gt_words']}\n"
            if 'total_ocr_words' in self.metrics:
                summary_text += f"OCR Detected Words: {self.metrics['total_ocr_words']}\n"
            if 'word_matches' in self.metrics:
                summary_text += f"Word Matches: {self.metrics['word_matches']}\n"
            
            # Quality assessment
            if 'overall_quality_score' in self.metrics:
                quality = self.metrics['overall_quality_score']
                summary_text += f"\nOverall Quality Score: {quality:.3f}\n"
                
                if quality > 0.8:
                    summary_text += "Status: Excellent ✓"
                elif quality > 0.7:
                    summary_text += "Status: Good ✓"
                elif quality > 0.6:
                    summary_text += "Status: Fair ⚠"
                else:
                    summary_text += "Status: Poor ✗"
            
            axes[1, 0].text(0.05, 0.95, summary_text, transform=axes[1, 0].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Plot 4: Quality Score Breakdown
            axes[1, 1].set_title('Quality Score Components', fontweight='bold', pad=15)
            
            # Create quality score breakdown
            if 'quality_factors' in self.metrics and len(self.metrics['quality_factors']) >= 5:
                quality_labels = ['Character\nAccuracy', 'Word\nAccuracy', 'Word F1\nScore', 'Confidence', 'Detection\nRate']
                quality_values = self.metrics['quality_factors'][:5]  # Take first 5 factors
                
                # Cap all quality factors between 0.0 and 1.0 for visualization
                quality_values = [max(0.0, min(val, 1.0)) for val in quality_values]
                
                bars = axes[1, 1].bar(quality_labels, quality_values, 
                                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#DDA0DD', '#FFD700'], alpha=0.8)
                axes[1, 1].set_ylabel('Quality Factor Score')
                axes[1, 1].set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, quality_values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                # Add overall quality score as horizontal line
                if 'overall_quality_score' in self.metrics:
                    overall_score = self.metrics['overall_quality_score']
                    axes[1, 1].axhline(y=overall_score, color='red', linestyle='-', alpha=0.8, 
                                      linewidth=3, label=f'Overall Quality: {overall_score:.3f}')
                    axes[1, 1].legend(fontsize=9)
                
                # Add threshold lines
                axes[1, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
                axes[1, 1].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.7)')
        
        elif has_region_metrics:
            # Fallback: Show basic region metrics if available
            axes[0, 0].set_title('Region Detection Analysis', fontweight='bold', pad=15)
            
            region_data = []
            region_labels = []
            region_colors = []
            
            if 'total_gt_regions' in self.metrics:
                region_data.append(self.metrics['total_gt_regions'])
                region_labels.append('Ground Truth\nRegions')
                region_colors.append('#4ECDC4')
            
            if 'total_ocr_regions' in self.metrics:
                region_data.append(self.metrics['total_ocr_regions'])
                region_labels.append('OCR Detected\nRegions')
                region_colors.append('#FF6B6B')
            
            if region_data:
                bars = axes[0, 0].bar(region_labels, region_data, color=region_colors, alpha=0.8)
                axes[0, 0].set_ylabel('Number of Regions')
                
                # Add value labels
                for bar, value in zip(bars, region_data):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Show basic OCR confidence if available
            if 'average_confidence' in self.metrics:
                axes[0, 1].bar(['OCR Confidence'], [self.metrics['average_confidence']], 
                              color='#45B7D1', alpha=0.8)
                axes[0, 1].set_ylabel('Confidence Score')
                axes[0, 1].set_ylim(0, 1)
                axes[0, 1].set_title('OCR Confidence', pad=15)
                
                # Add value label
                axes[0, 1].text(0.5, self.metrics['average_confidence'] + 0.01,
                               f'{self.metrics["average_confidence"]:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
            
            # Hide unused plots
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
        
        else:
            # No map-based metrics available
            axes[0, 0].text(0.5, 0.5, 'No map-based metrics available\n(No ground truth map provided)', 
                           ha='center', va='center', transform=axes[0, 0].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[0, 0].set_title('Map-Based Evaluation', pad=15)
            
            # Show basic image quality metrics if available
            if 'laplacian_variance' in self.metrics:
                axes[0, 1].bar(['Image Sharpness'], [min(self.metrics['laplacian_variance'] / 1000, 1.0)], 
                              color='#45B7D1', alpha=0.8)
                axes[0, 1].set_ylabel('Normalized Score')
                axes[0, 1].set_ylim(0, 1)
                axes[0, 1].set_title('Image Quality', pad=15)
            
            # Hide unused plots
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
        
                    # Adjust layout to eliminate white space
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)
        
        if output_path:
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Visualization saved to: {output_path}")
                if self.debug:
                    print(f"   Debug - Visualization file size: {os.path.getsize(output_path)} bytes")
            except Exception as e:
                print(f"Error saving visualization: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
        else:
            plt.show()

    def create_histogram_visualization(self, ground_truth_map: Dict[str, list], ocr_detected_map: Dict[str, list], image_path: str, output_path: str = None):
        """
        Create comprehensive histogram visualizations of evaluation data.
        
        Args:
            ground_truth_map: Dictionary containing ground truth data
            ocr_detected_map: Dictionary containing OCR detected data
            image_path: Path to the original image
            output_path: Path to save the histogram visualization
        """
        def cap_visualization_value(value, min_val=0.0, max_val=1.0):
            """Cap a value for visualization purposes and log if changed."""
            capped = max(min_val, min(value, max_val))
            if self.debug and value != capped:
                print(f"   Debug - Visualization value capped: {value:.3f} -> {capped:.3f}")
            return capped
        
        try:
            image_filename = os.path.basename(image_path)
            gt_entries = ground_truth_map.get(image_filename, [])
            ocr_entries = ocr_detected_map.get(image_filename, [])
            
            if self.debug:
                print(f"   Debug - Histogram visualization:")
                print(f"     Image filename: {image_filename}")
                print(f"     GT entries: {len(gt_entries)}")
                print(f"     OCR entries: {len(ocr_entries)}")
                if ocr_entries:
                    print(f"     Sample OCR entry: {ocr_entries[0]}")
            
            if not ocr_entries:
                print("No OCR entries found for histogram visualization")
                return
            
            # Create figure with subplots for different histogram types
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            image_filename = os.path.basename(image_path)
            fig.suptitle(f'OCR Evaluation: {image_filename} - Individual vs Overall Histogram Analysis', fontsize=18, fontweight='bold', y=0.98)
            
            # Row 1: Individual Bounding Box Analysis (One point per bounding box)
            # 1. Individual Bounding Box Confidence Analysis
            axes[0, 0].set_title('Individual Bounding Box Confidence (Scatter Plot)', fontweight='bold', pad=10)
            
            if len(ocr_entries) > 0:
                # Create individual confidence analysis for each bounding box
                individual_confidences = []
                individual_labels = []
                
                for i, entry in enumerate(ocr_entries[:10]):  # Show up to 10 bounding boxes
                    confidence = entry.get('confidence', 0.0)
                    capped_conf = cap_visualization_value(confidence)
                    individual_confidences.append(capped_conf)
                    individual_labels.append(f'Box {i+1}')
                
                # Create scatter plot for individual bounding box confidences
                x_positions = range(len(individual_confidences))
                axes[0, 0].scatter(x_positions, individual_confidences, 
                                  color='#4ECDC4', s=100, alpha=0.7, edgecolor='black', linewidth=2)
                axes[0, 0].set_ylabel('Confidence Score')
                # Set y-limits with padding to prevent overlap
                y_min = min(individual_confidences)
                y_max = max(individual_confidences)
                padding = max(0.05, (y_max - y_min) * 0.1)  # 10% padding, minimum 0.05
                axes[0, 0].set_ylim(max(0, y_min - padding), min(1, y_max + padding))
                axes[0, 0].set_xlabel('Bounding Box Index')
                axes[0, 0].set_xticks(x_positions)
                axes[0, 0].set_xticklabels(individual_labels)
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add value labels on points
                for i, value in enumerate(individual_confidences):
                    # Adjust label position to prevent overlap with title
                    label_y = value + padding * 0.5
                    if label_y > y_max + padding * 0.8:  # If label would be too close to top
                        label_y = value - padding * 0.5  # Place label below the point
                    axes[0, 0].text(i, label_y, f'{value:.3f}', 
                                   ha='center', va='bottom', fontweight='bold', fontsize=8)
                
                # Add mean line
                mean_conf = np.mean(individual_confidences)
                axes[0, 0].axhline(y=mean_conf, color='red', linestyle='--', 
                                 label=f'Mean: {mean_conf:.3f}')
                axes[0, 0].legend()
            else:
                axes[0, 0].text(0.5, 0.5, 'No OCR entries available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
            
            # 2. Individual Bounding Box Text Length Analysis
            axes[0, 1].set_title('Individual Bounding Box Text Length (Scatter Plot)', fontweight='bold', pad=10)
            
            if len(ocr_entries) > 0:
                # Create individual text length analysis for each bounding box
                individual_lengths = []
                individual_labels = []
                
                for i, entry in enumerate(ocr_entries[:10]):  # Show up to 10 bounding boxes
                    text_length = len(entry.get('text', ''))
                    individual_lengths.append(text_length)
                    individual_labels.append(f'Box {i+1}')
                
                # Create scatter plot for individual bounding box text lengths
                x_positions = range(len(individual_lengths))
                axes[0, 1].scatter(x_positions, individual_lengths, 
                                  color='#45B7D1', s=100, alpha=0.7, edgecolor='black', linewidth=2)
                axes[0, 1].set_ylabel('Text Length (characters)')
                # Set y-limits with padding to prevent overlap
                y_min = min(individual_lengths)
                y_max = max(individual_lengths)
                padding = max(1, (y_max - y_min) * 0.1)  # 10% padding, minimum 1
                axes[0, 1].set_ylim(max(0, y_min - padding), y_max + padding)
                axes[0, 1].set_xlabel('Bounding Box Index')
                axes[0, 1].set_xticks(x_positions)
                axes[0, 1].set_xticklabels(individual_labels)
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add value labels on points
                for i, value in enumerate(individual_lengths):
                    # Adjust label position to prevent overlap with title
                    label_y = value + padding * 0.5
                    if label_y > y_max + padding * 0.8:  # If label would be too close to top
                        label_y = value - padding * 0.5  # Place label below the point
                    axes[0, 1].text(i, label_y, f'{value}', 
                                   ha='center', va='bottom', fontweight='bold', fontsize=8)
                
                # Add mean line
                mean_length = np.mean(individual_lengths)
                axes[0, 1].axhline(y=mean_length, color='red', linestyle='--', 
                                 label=f'Mean: {mean_length:.1f}')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No OCR entries available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # 3. Individual Bounding Box Area Analysis
            axes[0, 2].set_title('Individual Bounding Box Area (Scatter Plot)', fontweight='bold', pad=10)
            
            if len(ocr_entries) > 0:
                # Create individual area analysis for each bounding box
                individual_areas = []
                individual_labels = []
                
                for i, entry in enumerate(ocr_entries[:10]):  # Show up to 10 bounding boxes
                    bbox = entry.get('bbox', [])
                    if len(bbox) == 4:
                        area = bbox[2] * bbox[3]  # width * height in percentage
                        individual_areas.append(area)
                        individual_labels.append(f'Box {i+1}')
                
                if individual_areas:
                    # Create scatter plot for individual bounding box areas
                    x_positions = range(len(individual_areas))
                    axes[0, 2].scatter(x_positions, individual_areas, 
                                      color='#FF6B6B', s=100, alpha=0.7, edgecolor='black', linewidth=2)
                    axes[0, 2].set_ylabel('Area (% of image)')
                    # Set y-limits with padding to prevent overlap, but handle small values properly
                    y_min = min(individual_areas)
                    y_max = max(individual_areas)
                    padding = max(0.001, (y_max - y_min) * 0.1)  # 10% padding, minimum 0.001
                    axes[0, 2].set_ylim(max(0, y_min - padding), y_max + padding)
                    axes[0, 2].set_xlabel('Bounding Box Index')
                    axes[0, 2].set_xticks(x_positions)
                    axes[0, 2].set_xticklabels(individual_labels)
                    axes[0, 2].grid(True, alpha=0.3)
                    
                    # Add value labels on points
                    for i, value in enumerate(individual_areas):
                        # Adjust label position to prevent overlap with title
                        label_y = value + padding * 0.5
                        if label_y > y_max + padding * 0.8:  # If label would be too close to top
                            label_y = value - padding * 0.5  # Place label below the point
                        axes[0, 2].text(i, label_y, f'{value:.4f}', 
                                       ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
                    # Add mean line
                    mean_area = np.mean(individual_areas)
                    axes[0, 2].axhline(y=mean_area, color='red', linestyle='--', 
                                     label=f'Mean: {mean_area:.4f}')
                    axes[0, 2].legend()
                else:
                    axes[0, 2].text(0.5, 0.5, 'No valid bounding box data', 
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
            else:
                axes[0, 2].text(0.5, 0.5, 'No OCR entries available', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
            
            # Row 2: Overall Histogram Analysis (Distribution across all bounding boxes)
            # 4. Overall Confidence Distribution (Histogram)
            axes[1, 0].set_title('Overall Confidence Distribution (Histogram)', fontweight='bold', pad=10)
            confidences = [entry.get('confidence', 0.0) for entry in ocr_entries]
            if confidences:
                # Cap confidence values between 0.0 and 1.0
                capped_confidences = [cap_visualization_value(conf) for conf in confidences]
                
                axes[1, 0].hist(capped_confidences, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
                mean_conf = np.mean(capped_confidences)
                median_conf = np.median(capped_confidences)
                axes[1, 0].axvline(mean_conf, color='red', linestyle='--', 
                                 label=f'Mean: {mean_conf:.3f}')
                axes[1, 0].axvline(median_conf, color='orange', linestyle='--', 
                                 label=f'Median: {median_conf:.3f}')
                axes[1, 0].set_xlabel('Confidence Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_xlim(0, 1)  # Ensure x-axis is capped
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No confidence data available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # 5. Overall Text Length Distribution (Histogram)
            axes[1, 1].set_title('Overall Text Length Distribution (Histogram)', fontweight='bold', pad=10)
            text_lengths = [len(entry.get('text', '')) for entry in ocr_entries]
            if text_lengths:
                # Cap text lengths to reasonable range for visualization, but log when capping occurs
                max_length = min(max(text_lengths), 100)  # Cap at 100 characters
                capped_lengths = []
                capped_count = 0
                for length in text_lengths:
                    if length > 100:
                        capped_count += 1
                    capped_lengths.append(min(length, max_length))
                
                if self.debug and capped_count > 0:
                    print(f"   Debug - Text length capping: {capped_count} entries capped at 100 characters")
                
                axes[1, 1].hist(capped_lengths, bins=20, alpha=0.7, color='#45B7D1', edgecolor='black')
                mean_length = np.mean(capped_lengths)
                median_length = np.median(capped_lengths)
                axes[1, 1].axvline(mean_length, color='red', linestyle='--', 
                                 label=f'Mean: {mean_length:.1f}')
                axes[1, 1].axvline(median_length, color='orange', linestyle='--', 
                                 label=f'Median: {median_length:.1f}')
                axes[1, 1].set_xlabel('Text Length (characters)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No text length data available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            # 6. Overall Area Distribution (Histogram)
            axes[1, 2].set_title('Overall Area Distribution (Histogram)', fontweight='bold', pad=10)
            areas = []
            
            if self.debug:
                print(f"   Debug - Processing {len(ocr_entries)} OCR entries for area distribution")
            
            for entry in ocr_entries:
                bbox = entry.get('bbox', [])
                if len(bbox) == 4:
                    # Calculate area in percentage (width_percent * height_percent)
                    area = bbox[2] * bbox[3]  # width * height in percentage
                    areas.append(area)
                    if self.debug:
                        print(f"   Debug - Entry bbox: {bbox}, area: {area:.6f}")
                else:
                    if self.debug:
                        print(f"   Debug - Entry missing or invalid bbox: {bbox}")
            
            if areas:
                # Don't cap the areas for histogram - let them show the actual distribution
                axes[1, 2].hist(areas, bins=min(20, len(areas)), alpha=0.7, color='#FF6B6B', edgecolor='black')
                mean_area = np.mean(areas)
                median_area = np.median(areas)
                axes[1, 2].axvline(mean_area, color='red', linestyle='--', 
                                 label=f'Mean: {mean_area:.4f}')
                axes[1, 2].axvline(median_area, color='orange', linestyle='--', 
                                 label=f'Median: {median_area:.4f}')
                axes[1, 2].set_xlabel('Area (% of image)')
                axes[1, 2].set_ylabel('Frequency')
                # Don't set xlim to 0,1 since areas are in percentage and very small
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
                
                if self.debug:
                    print(f"   Debug - Area distribution: min={min(areas):.6f}, max={max(areas):.6f}, mean={mean_area:.6f}")
            else:
                if self.debug:
                    print(f"   Debug - No valid areas found for histogram")
                axes[1, 2].text(0.5, 0.5, 'No area data available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
            
            # Adjust layout to prevent title overlap
            plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95, hspace=0.4, wspace=0.25)
            
            if output_path:
                try:
                    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"Histogram visualization saved to: {output_path}")
                    if self.debug:
                        print(f"   Debug - Histogram file size: {os.path.getsize(output_path)} bytes")
                        print(f"   Debug - Histogram visualization completed successfully")
                except Exception as e:
                    print(f"Error saving histogram visualization: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
            else:
                plt.show()
                if self.debug:
                    print(f"   Debug - Histogram visualization displayed")
                
        except Exception as e:
            print(f"Error creating histogram visualization: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='OCR Image Quality Evaluation Tool')
    parser.add_argument('image_path', help='Path to the image to evaluate')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    parser.add_argument('--visualization', '-v', help='Output file for visualization (PNG)')
    parser.add_argument('--output-dir', '-d', help='Output directory for all files (will create dated folder if not specified)')
    parser.add_argument('--ground-truth', '-g', help='Ground truth text for Levenshtein distance calculation')
    parser.add_argument('--tolerance', '-t', type=int, default=5, help='Bounding box matching tolerance in pixels (default: 5)')
    parser.add_argument('--levenshtein-threshold', '-l', type=int, default=1, help='Maximum Levenshtein distance allowed for word matching (default: 1)')
    parser.add_argument('--matching-mode', '-m', choices=['text_aware', 'position_focused'], default='text_aware', 
                       help='Matching strategy: text_aware (prioritize text similarity) or position_focused (prioritize bounding box similarity) (default: text_aware)')
    parser.add_argument('--no-display', action='store_true', help='Don\'t display results')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug output')
    
    args = parser.parse_args()
    
    # Create evaluation tool with debug mode and matching strategy
    debug_mode = args.debug or (not args.no_debug)  # Default to debug mode unless explicitly disabled
    evaluator = OCREvaluationTool(debug=debug_mode, matching_mode=args.matching_mode, levenshtein_threshold=args.levenshtein_threshold)
    
    # Load ground truth map
    ground_truth_from_map = None
    import os
    import json
    gt_map_path = os.path.join(os.path.dirname(__file__), 'Ground_Truth_Maps', 'ground_truth_map.json')
    if os.path.exists(gt_map_path):
        with open(gt_map_path, 'r') as f:
            gt_map = json.load(f)
        image_filename = os.path.basename(args.image_path)
        ground_truth_entries = gt_map.get(image_filename, [])
        
        if ground_truth_entries:
            # Extract all text from the ground truth entries
            ground_truth_texts = [entry['text'] for entry in ground_truth_entries if 'text' in entry]
            ground_truth_text = ' '.join(ground_truth_texts)
            
            # Extract bounding boxes for potential region-specific analysis
            ground_truth_bboxes = [entry['bbox'] for entry in ground_truth_entries if 'bbox' in entry]
            
            # Store ground truth entries in evaluator for visualization
            evaluator.ground_truth_entries = ground_truth_entries
            
            print(f"Using ground truth from map for {image_filename}.")
            print(f"Found {len(ground_truth_entries)} text regions with {len(ground_truth_texts)} text entries.")
            print(f"Ground truth text: '{ground_truth_text}'")
            print(f"Bounding boxes available: {len(ground_truth_bboxes)}")
        else:
            ground_truth_text = None
            ground_truth_bboxes = None
            evaluator.ground_truth_entries = []
            print(f"No ground truth found in map for {image_filename}.")
    else:
        ground_truth_text = None
        ground_truth_bboxes = None
        print(f"Ground truth map not found at: {gt_map_path}")
    
    try:
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # Create default dated directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"Run_{timestamp}"
        
        # Get absolute path to ensure we're in the right location
        output_dir = os.path.abspath(output_dir)
        print(f"Output directory (absolute): {output_dir}")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Evaluate image with timing
        import time
        start_time = time.time()
        
        metrics = evaluator.evaluate_image(args.image_path, ground_truth_text, ground_truth_bboxes, output_dir, tolerance=args.tolerance)
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        print(f"\n⏱️  Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Clear GPU memory
        evaluator.clear_gpu_memory()
        
        # Print results
        if not args.no_display:
            evaluator.print_results()
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(output_dir, "evaluation_results.json")
        
        print(f"Saving results to: {output_path}")
        evaluator.save_results(output_path)
        print(f"Results saved to: {output_path}")
        
        # Create visualization
        if args.visualization:
            viz_path = args.visualization
        else:
            viz_path = os.path.join(output_dir, "evaluation_visualization.png")
        
        print(f"Saving visualization to: {viz_path}")
        if not args.no_display:
            evaluator.create_visualization(args.image_path, viz_path)
            print(f"Visualization saved to: {viz_path}")
        
        # Verify files were created
        if os.path.exists(output_path):
            print(f"✓ JSON file confirmed: {output_path}")
        else:
            print(f"✗ JSON file not found: {output_path}")
            
        if os.path.exists(viz_path):
            print(f"✓ Visualization file confirmed: {viz_path}")
        else:
            print(f"✗ Visualization file not found: {viz_path}")
        
        print(f"\nAll outputs saved to: {output_dir}/")
        print(f"Files created:")
        print(f"- {output_dir}/evaluation_results.json (metrics data)")
        print(f"- {output_dir}/evaluation_visualization.png (visual analysis)")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 