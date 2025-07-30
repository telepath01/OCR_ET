import json
import os
from typing import Dict, List, Tuple, Union

class GroundTruthProcessor:
    """
    Process Label Studio exports and extract ground truth text with bounding boxes for OCR evaluation.
    """
    
    def __init__(self, export_file_path: str):
        """
        Initialize with Label Studio export file.
        
        Args:
            export_file_path: Path to the Label Studio JSON export file
        """
        self.export_file_path = export_file_path
        self.ground_truth_data = self._load_export_data()
    
    def _load_export_data(self) -> List[Dict]:
        """Load the Label Studio export data."""
        with open(self.export_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_ground_truth(self) -> Dict[str, List[Dict]]:
        """
        Extract ground truth text with bounding boxes for each image.
        
        Returns:
            Dictionary mapping image filenames to list of ground truth objects
            Each object contains: {'text': str, 'bbox': [x, y, width, height]}
        """
        ground_truth_map = {}
        
        for task in self.ground_truth_data:
            # Get the image filename
            image_filename = self._get_image_filename(task)
            
            # Extract the ground truth text and bounding boxes from annotations
            ground_truth_objects = self._extract_text_and_bbox_from_annotations(task)
            
            if ground_truth_objects:
                ground_truth_map[image_filename] = ground_truth_objects
        
        return ground_truth_map
    
    def extract_ground_truth_text_only(self) -> Dict[str, str]:
        """
        Extract ground truth text only (for backward compatibility).
        
        Returns:
            Dictionary mapping image filenames to ground truth text
        """
        ground_truth_map = {}
        
        for task in self.ground_truth_data:
            # Get the image filename
            image_filename = self._get_image_filename(task)
            
            # Extract the ground truth text from annotations
            ground_truth_text = self._extract_text_from_annotations(task)
            
            if ground_truth_text:
                ground_truth_map[image_filename] = ground_truth_text
        
        return ground_truth_map
    
    def _get_image_filename(self, task: Dict) -> str:
        """Extract image filename from task data."""
        # Check for file_upload field first (most common in Label Studio)
        if 'file_upload' in task:
            file_upload = task['file_upload']
            # Extract the actual filename from the file_upload path
            # Format is usually: "hash-filename.jpg"
            if '-' in file_upload:
                # Split by '-' and take everything after the first '-'
                parts = file_upload.split('-', 1)
                if len(parts) > 1:
                    return parts[1]  # Return the actual filename
            return os.path.basename(file_upload)
        
        # If using local file storage
        elif 'data' in task and 'image' in task['data']:
            image_path = task['data']['image']
            return os.path.basename(image_path)
        
        # If using cloud storage
        elif 'data' in task and 'image' in task['data']:
            image_url = task['data']['image']
            # Extract filename from URL
            return os.path.basename(image_url.split('?')[0])
        
        # If using file upload
        elif 'data' in task and 'file' in task['data']:
            file_path = task['data']['file']
            return os.path.basename(file_path)
        
        else:
            # Fallback: use task ID
            return f"task_{task.get('id', 'unknown')}.jpg"
    
    def _extract_text_and_bbox_from_annotations(self, task: Dict) -> List[Dict]:
        """Extract ground truth text and bounding boxes from task annotations."""
        if 'annotations' not in task or not task['annotations']:
            return []
        
        # Get the first annotation (assuming single annotator or consensus)
        annotation = task['annotations'][0]
        
        if 'result' not in annotation:
            return []
        
        # Extract text and bounding boxes from different annotation types
        extracted_objects = []
        
        for result in annotation['result']:
            # Text annotation with bounding box (TextArea, TextInput, etc.)
            if result.get('type') in ['textarea', 'textinput', 'text']:
                if 'value' in result and 'text' in result['value']:
                    text_value = result['value']['text']
                    # Handle both string and list formats
                    if isinstance(text_value, list):
                        # If it's a list, join the elements
                        text_str = ' '.join(str(item) for item in text_value)
                    else:
                        # If it's a string, use it directly
                        text_str = str(text_value)
                    
                    # Extract bounding box if available
                    bbox = self._extract_bbox_from_result(result)
                    
                    if text_str.strip():
                        extracted_objects.append({
                            'text': text_str.strip(),
                            'bbox': bbox
                        })
            
            # Rectangle labels with text and bounding box
            elif result.get('type') == 'rectanglelabels':
                if 'value' in result:
                    value = result['value']
                    # Extract labels/text
                    text_str = ""
                    if 'labels' in value:
                        labels = value['labels']
                        if labels:
                            text_str = ' '.join(labels)
                    
                    # Extract bounding box
                    bbox = self._extract_bbox_from_result(result)
                    
                    if text_str.strip():
                        extracted_objects.append({
                            'text': text_str.strip(),
                            'bbox': bbox
                        })
            
            # OCR annotation with bounding box
            elif result.get('type') == 'textarea':
                if 'value' in result and 'text' in result['value']:
                    text_value = result['value']['text']
                    # Handle both string and list formats
                    if isinstance(text_value, list):
                        # If it's a list, join the elements
                        text_str = ' '.join(str(item) for item in text_value)
                    else:
                        # If it's a string, use it directly
                        text_str = str(text_value)
                    
                    # Extract bounding box if available
                    bbox = self._extract_bbox_from_result(result)
                    
                    if text_str.strip():
                        extracted_objects.append({
                            'text': text_str.strip(),
                            'bbox': bbox
                        })
            
            # Handle other annotation types that might contain text and bounding boxes
            elif 'value' in result:
                value = result['value']
                # Check for text in various possible locations
                for key in ['text', 'label', 'labels', 'value']:
                    if key in value:
                        text_value = value[key]
                        if isinstance(text_value, list):
                            text_str = ' '.join(str(item) for item in text_value)
                        else:
                            text_str = str(text_value)
                        
                        if text_str.strip():
                            # Extract bounding box if available
                            bbox = self._extract_bbox_from_result(result)
                            
                            extracted_objects.append({
                                'text': text_str.strip(),
                                'bbox': bbox
                            })
        
        return extracted_objects
    
    def _extract_bbox_from_result(self, result: Dict) -> Union[List[float], None]:
        """Extract bounding box coordinates from a result object."""
        if 'value' not in result:
            return None
        
        value = result['value']
        
        # Check for different bounding box formats in Label Studio
        # Format 1: x, y, width, height (percentage)
        if all(key in value for key in ['x', 'y', 'width', 'height']):
            return [
                float(value['x']),
                float(value['y']),
                float(value['width']),
                float(value['height'])
            ]
        
        # Format 2: x, y, width, height (absolute pixels)
        elif all(key in value for key in ['x', 'y', 'width', 'height']):
            return [
                float(value['x']),
                float(value['y']),
                float(value['width']),
                float(value['height'])
            ]
        
        # Format 3: coordinates as a list [x, y, width, height]
        elif 'coordinates' in value:
            coords = value['coordinates']
            if isinstance(coords, list) and len(coords) >= 4:
                return [float(coord) for coord in coords[:4]]
        
        # Format 4: bbox field
        elif 'bbox' in value:
            bbox = value['bbox']
            if isinstance(bbox, list) and len(bbox) >= 4:
                return [float(coord) for coord in bbox[:4]]
        
        return None

    def _extract_text_from_annotations(self, task: Dict) -> str:
        """Extract ground truth text from task annotations (for backward compatibility)."""
        ground_truth_objects = self._extract_text_and_bbox_from_annotations(task)
        texts = [obj['text'] for obj in ground_truth_objects]
        return ' '.join(texts).strip()
    
    def save_ground_truth_map(self, output_path: str):
        """Save the ground truth mapping with bounding boxes to a JSON file."""
        ground_truth_map = self.extract_ground_truth()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_map, f, indent=2, ensure_ascii=False)
        
        print(f"Ground truth map with bounding boxes saved to: {output_path}")
        print(f"Total images with ground truth: {len(ground_truth_map)}")
        
        # Count total text regions
        total_regions = sum(len(regions) for regions in ground_truth_map.values())
        print(f"Total text regions: {total_regions}")
        
        return ground_truth_map
    
    def save_ground_truth_text_only(self, output_path: str):
        """Save the ground truth mapping (text only) to a JSON file (for backward compatibility)."""
        ground_truth_map = self.extract_ground_truth_text_only()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_map, f, indent=2, ensure_ascii=False)
        
        print(f"Ground truth map (text only) saved to: {output_path}")
        print(f"Total images with ground truth: {len(ground_truth_map)}")
        
        return ground_truth_map
    
    def get_ground_truth_for_image(self, image_filename: str) -> List[Dict]:
        """Get ground truth text and bounding boxes for a specific image."""
        ground_truth_map = self.extract_ground_truth()
        return ground_truth_map.get(image_filename, [])
    
    def get_ground_truth_text_for_image(self, image_filename: str) -> str:
        """Get ground truth text only for a specific image (for backward compatibility)."""
        ground_truth_map = self.extract_ground_truth_text_only()
        return ground_truth_map.get(image_filename, "")
    
    def debug_annotation_structure(self, max_examples: int = 3):
        """Debug method to inspect the structure of annotations."""
        print(f"Total tasks in export: {len(self.ground_truth_data)}")
        
        for i, task in enumerate(self.ground_truth_data[:max_examples]):
            print(f"\n--- Task {i+1} ---")
            print(f"Task ID: {task.get('id', 'N/A')}")
            
            # Show data structure
            if 'data' in task:
                print("Data keys:", list(task['data'].keys()))
                for key, value in task['data'].items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
            
            # Show annotations structure
            if 'annotations' in task and task['annotations']:
                annotation = task['annotations'][0]
                print(f"Annotation ID: {annotation.get('id', 'N/A')}")
                print(f"Result count: {len(annotation.get('result', []))}")
                
                for j, result in enumerate(annotation.get('result', [])[:2]):  # Show first 2 results
                    print(f"  Result {j+1}:")
                    print(f"    Type: {result.get('type', 'N/A')}")
                    print(f"    Value keys: {list(result.get('value', {}).keys())}")
                    
                    # Show value structure
                    if 'value' in result:
                        for key, value in result['value'].items():
                            if key == 'text':
                                print(f"    {key}: {value} (type: {type(value)})")
                            else:
                                print(f"    {key}: {value}")

# Example usage
if __name__ == "__main__":
    # Initialize processor with your Label Studio export
    processor = GroundTruthProcessor("label_studio_export.json")
    
    # First, let's debug the structure to understand your data
    print("=== DEBUGGING ANNOTATION STRUCTURE ===")
    processor.debug_annotation_structure()
    
    print("\n=== EXTRACTING GROUND TRUTH WITH BOUNDING BOXES ===")
    # Save ground truth mapping with bounding boxes
    ground_truth_map = processor.save_ground_truth_map("ground_truth_map_with_bbox.json")
    
    # Print some examples
    print("\nExample ground truth mappings with bounding boxes:")
    for i, (filename, regions) in enumerate(list(ground_truth_map.items())[:3]):
        print(f"\n{filename}:")
        for j, region in enumerate(regions[:2]):  # Show first 2 regions per image
            print(f"  Region {j+1}:")
            print(f"    Text: {region['text']}")
            print(f"    BBox: {region['bbox']}")
    
    # Also save text-only version for backward compatibility
    print("\n=== SAVING TEXT-ONLY VERSION ===")
    processor.save_ground_truth_text_only("ground_truth_map_text_only.json")