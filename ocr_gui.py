import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QSpacerItem, QSizePolicy, QFileDialog, QMessageBox, QMainWindow, QScrollArea, QCheckBox, QGroupBox, QSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import os
import traceback
import io
import matplotlib.pyplot as plt
import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

class ImageDisplayWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle('OCR Evaluation Results - Visualization')
        self.setMinimumSize(1000, 800)
        self.setGeometry(100, 100, 1200, 900)  # Set initial position and size
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        # Create scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        # Create label for image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(self.image_label)
        # Load and display image from file
        print(f"Debug - Attempting to load image: {image_path}")
        print(f"Debug - File exists: {os.path.exists(image_path)}")
        
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            print(f"Debug - Image loaded successfully, size: {pixmap.width()}x{pixmap.height()}")
            scaled_pixmap = pixmap.scaled(scroll_area.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            print(f"Debug - Image displayed successfully")
            
            # Add a status label
            status_label = QLabel("✓ Visualization loaded successfully!")
            status_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold; padding: 10px;")
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(status_label)
        else:
            print(f"Debug - Failed to load image: {image_path}")
            self.image_label.setText(f"Failed to load image:\n{image_path}")
            
            # Add error status label
            error_label = QLabel("✗ Failed to load visualization!")
            error_label.setStyleSheet("color: red; font-size: 14px; font-weight: bold; padding: 10px;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        # Exit button
        exit_btn = QPushButton('Exit')
        exit_btn.setStyleSheet('font-size: 14px; padding: 8px 16px;')
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(exit_btn)
        layout.addLayout(button_layout)

class MultiImageDisplayWindow(QMainWindow):
    def __init__(self, main_image_path, bbox_image_path):
        super().__init__()
        self.setWindowTitle('OCR Evaluation Results - Multiple Visualizations')
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create horizontal layout for two images
        image_layout = QHBoxLayout()
        
        # Left side - Main evaluation visualization
        left_group = QGroupBox('Evaluation Metrics Visualization')
        left_group.setStyleSheet('font-size: 14px; font-weight: bold; margin: 5px; padding: 5px;')
        left_layout = QVBoxLayout()
        
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_image_label = QLabel()
        left_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_pixmap = QPixmap(main_image_path)
        if not left_pixmap.isNull():
            scaled_pixmap = left_pixmap.scaled(left_scroll.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            left_image_label.setPixmap(scaled_pixmap)
        else:
            left_image_label.setText("Failed to load main visualization")
        left_scroll.setWidget(left_image_label)
        left_layout.addWidget(left_scroll)
        left_group.setLayout(left_layout)
        image_layout.addWidget(left_group)
        
        # Right side - Bounding box visualization
        right_group = QGroupBox('Ground Truth vs OCR Results')
        right_group.setStyleSheet('font-size: 14px; font-weight: bold; margin: 5px; padding: 5px;')
        right_layout = QVBoxLayout()
        
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_image_label = QLabel()
        right_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_pixmap = QPixmap(bbox_image_path)
        if not right_pixmap.isNull():
            scaled_pixmap = right_pixmap.scaled(right_scroll.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            right_image_label.setPixmap(scaled_pixmap)
        else:
            right_image_label.setText("Failed to load bounding box visualization")
        right_scroll.setWidget(right_image_label)
        right_layout.addWidget(right_scroll)
        right_group.setLayout(right_layout)
        image_layout.addWidget(right_group)
        
        layout.addLayout(image_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Exit button
        exit_btn = QPushButton('Exit')
        exit_btn.setStyleSheet('font-size: 14px; padding: 8px 16px;')
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(exit_btn)
        layout.addLayout(button_layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OCR Evaluation Tool')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.image_path = None
        self.ground_truth_path = None
        self.batch_folder_path = None  # For batch processing
        self.evaluate_button = None  # Will be set in show_ocr_evaluation_ui
        self.bbox_tolerance = 1
        self.init_ui()

    def init_ui(self):
        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.logo_label.setStyleSheet('margin-bottom: 10px;')
        
        # Load and display logo
        logo_path = os.path.join(os.path.dirname(__file__), 'Logo', 'OCR_Logo.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                # Scale logo to reasonable size (max width 200px, maintain aspect ratio)
                scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.logo_label.setPixmap(scaled_pixmap)
            else:
                print(f"Debug - Failed to load logo: {logo_path}")
        else:
            print(f"Debug - Logo file not found: {logo_path}")
        
        self.layout.addWidget(self.logo_label)

        # Main Title
        self.main_title = QLabel('OCR Evaluator')
        self.main_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.main_title.setStyleSheet('font-size: 28px; font-weight: bold; margin-bottom: 10px; color: #2c3e50;')
        self.layout.addWidget(self.main_title)

        # Subtitle
        self.subtitle = QLabel('Please Select an Option')
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.subtitle.setStyleSheet('font-size: 16px; font-weight: normal; margin-bottom: 25px; color: #7f8c8d;')
        self.layout.addWidget(self.subtitle)

        # Grid 1: Dropdown Selection
        grid1_group = QGroupBox('Type of Evaluation')
        grid1_group.setStyleSheet('QGroupBox { font-size: 14px; font-weight: bold; margin-top: 10px; padding-top: 10px; border: 2px solid #bdc3c7; border-radius: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }')
        grid1_layout = QVBoxLayout()
        
        # Dropdown
        self.combo = QComboBox()
        self.combo.addItems(['Single Image Evaluation', 'Batch Image Evaluation', 'Ground Truth Processor'])
        self.combo.setStyleSheet('font-size: 16px; padding: 8px; border: 1px solid #bdc3c7; border-radius: 4px;')
        self.combo.currentIndexChanged.connect(self.on_option_changed)
        grid1_layout.addWidget(self.combo)
        
        grid1_group.setLayout(grid1_layout)
        self.layout.addWidget(grid1_group)

        # Grid 2: File Selection (Two Columns)
        self.grid2_group = QGroupBox('File Selection')
        self.grid2_group.setStyleSheet('QGroupBox { font-size: 14px; font-weight: bold; margin-top: 15px; padding-top: 10px; border: 2px solid #bdc3c7; border-radius: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }')
        grid2_layout = QHBoxLayout()
        
        # Left Column: Select Image
        left_column = QVBoxLayout()
        self.image_label = QLabel('Select Image:')
        self.image_label.setStyleSheet('font-size: 15px; font-weight: bold; margin-bottom: 5px;')
        left_column.addWidget(self.image_label)
        
        self.image_path_label = QLabel('No image selected')
        self.image_path_label.setStyleSheet('font-size: 12px; color: gray; margin-bottom: 5px;')
        left_column.addWidget(self.image_path_label)
        
        self.image_browse_btn = QPushButton('Browse Image')
        self.image_browse_btn.setStyleSheet('font-size: 14px; padding: 8px 16px; background-color: #3498db; color: white; border: none; border-radius: 4px;')
        self.image_browse_btn.clicked.connect(self.browse_image)
        left_column.addWidget(self.image_browse_btn)
        
        # Right Column: Select Ground Truth Map
        right_column = QVBoxLayout()
        self.gt_label = QLabel('Select Ground Truth Map:')
        self.gt_label.setStyleSheet('font-size: 15px; font-weight: bold; margin-bottom: 5px;')
        right_column.addWidget(self.gt_label)
        
        self.gt_path_label = QLabel('No ground truth map selected')
        self.gt_path_label.setStyleSheet('font-size: 12px; color: gray; margin-bottom: 5px;')
        right_column.addWidget(self.gt_path_label)
        
        self.gt_browse_btn = QPushButton('Browse Ground Truth')
        self.gt_browse_btn.setStyleSheet('font-size: 14px; padding: 8px 16px; background-color: #3498db; color: white; border: none; border-radius: 4px;')
        self.gt_browse_btn.clicked.connect(self.browse_ground_truth)
        right_column.addWidget(self.gt_browse_btn)
        
        # Add columns to grid2 layout
        grid2_layout.addLayout(left_column)
        grid2_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))  # Spacing between columns
        grid2_layout.addLayout(right_column)
        
        self.grid2_group.setLayout(grid2_layout)
        self.layout.addWidget(self.grid2_group)

        # Grid 3: Additional Options
        self.grid3_group = QGroupBox('Additional Options')
        self.grid3_group.setStyleSheet('QGroupBox { font-size: 14px; font-weight: bold; margin-top: 15px; padding-top: 10px; border: 2px solid #bdc3c7; border-radius: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }')
        grid3_layout = QVBoxLayout()
        
        # Sample size row (initially hidden, shown for batch processing)
        self.sample_size_widget = QWidget()
        self.sample_size_row = QHBoxLayout()
        self.sample_size_label = QLabel('Number of images to process:')
        self.sample_size_label.setStyleSheet('font-size: 13px;')
        self.sample_size_row.addWidget(self.sample_size_label)
        self.sample_size_spinbox = QSpinBox()
        self.sample_size_spinbox.setMinimum(1)
        self.sample_size_spinbox.setMaximum(1)  # Will be updated after folder selection
        self.sample_size_spinbox.setValue(1)
        self.sample_size_spinbox.setEnabled(False)
        self.sample_size_spinbox.valueChanged.connect(self.validate_batch_file_selections)
        self.sample_size_row.addWidget(self.sample_size_spinbox)
        self.sample_size_row.addStretch()  # Add stretch to push controls to the left
        self.sample_size_widget.setLayout(self.sample_size_row)
        grid3_layout.addWidget(self.sample_size_widget)
        
        # Bounding box tolerance row
        tolerance_widget = QWidget()
        tolerance_row = QHBoxLayout()
        tolerance_label = QLabel('Bounding Box Tolerance:')
        tolerance_label.setStyleSheet('font-size: 13px;')
        tolerance_row.addWidget(tolerance_label)
        self.bbox_tolerance_spinbox = QSpinBox()
        self.bbox_tolerance_spinbox.setMinimum(0)
        self.bbox_tolerance_spinbox.setMaximum(10)
        self.bbox_tolerance_spinbox.setValue(self.bbox_tolerance)
        self.bbox_tolerance_spinbox.valueChanged.connect(lambda value: setattr(self, 'bbox_tolerance', value))
        tolerance_row.addWidget(self.bbox_tolerance_spinbox)
        tolerance_row.addStretch()  # Add stretch to push controls to the left
        tolerance_widget.setLayout(tolerance_row)
        grid3_layout.addWidget(tolerance_widget)
        
        # Levenshtein threshold row
        levenshtein_widget = QWidget()
        levenshtein_row = QHBoxLayout()
        levenshtein_label = QLabel('Levenshtein Threshold:')
        levenshtein_label.setStyleSheet('font-size: 13px;')
        levenshtein_row.addWidget(levenshtein_label)
        self.levenshtein_threshold_spinbox = QSpinBox()
        self.levenshtein_threshold_spinbox.setMinimum(0)
        self.levenshtein_threshold_spinbox.setMaximum(5)
        self.levenshtein_threshold_spinbox.setValue(1)  # Default value
        self.levenshtein_threshold_spinbox.setToolTip('Maximum character differences allowed for word matching (0 = exact match, 1 = allow 1 character difference, etc.)')
        levenshtein_row.addWidget(self.levenshtein_threshold_spinbox)
        levenshtein_row.addStretch()  # Add stretch to push controls to the left
        levenshtein_widget.setLayout(levenshtein_row)
        grid3_layout.addWidget(levenshtein_widget)
        
        self.grid3_group.setLayout(grid3_layout)
        self.layout.addWidget(self.grid3_group)

        # Placeholder for dynamic widgets (for evaluate buttons and other dynamic content)
        self.dynamic_widget = QWidget()
        self.dynamic_layout = QVBoxLayout()
        self.dynamic_layout.setContentsMargins(20, 10, 20, 10)  # Left, Top, Right, Bottom margins
        self.dynamic_layout.setSpacing(12)  # Add spacing between widgets
        self.dynamic_widget.setLayout(self.dynamic_layout)
        self.layout.addWidget(self.dynamic_widget)

        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Bottom row for Quit button
        bottom_row = QHBoxLayout()
        bottom_row.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        quit_btn = QPushButton('Quit')
        quit_btn.setStyleSheet('font-size: 14px; padding: 8px 24px;')
        quit_btn.clicked.connect(self.close)
        bottom_row.addWidget(quit_btn)
        self.layout.addLayout(bottom_row)

        self.setLayout(self.layout)
        self.on_option_changed(0)  # Set initial state

    def clear_dynamic_layout(self):
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            # Also clear layouts to ensure all child widgets are removed
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    layout_item = layout.takeAt(0)
                    layout_widget = layout_item.widget()
                    if layout_widget is not None:
                        layout_widget.deleteLater()
        # Reset evaluate button reference when clearing layout
        self.evaluate_button = None
        self.adjustSize()

    def on_option_changed(self, index):
        self.clear_dynamic_layout()
        if self.combo.currentText() == 'Ground Truth Processor':
            self.show_ground_truth_ui()
        elif self.combo.currentText() == 'Single Image Evaluation':
            self.show_ocr_evaluation_ui()
        elif self.combo.currentText() == 'Batch Image Evaluation':
            self.show_batch_ocr_evaluation_ui()

    def show_ground_truth_ui(self):
        # Hide Grid 2 and Grid 3 since Ground Truth Processor doesn't need file selection or additional options
        self.grid2_group.setVisible(False)
        self.grid3_group.setVisible(False)
        
        # Label
        label = QLabel('Select Label Studio export JSON file:')
        label.setStyleSheet('font-size: 15px;')
        self.dynamic_layout.addWidget(label)

        # Browse button
        browse_btn = QPushButton('Browse')
        browse_btn.setStyleSheet('font-size: 14px; padding: 6px 18px;')
        browse_btn.clicked.connect(self.browse_export_file)
        self.dynamic_layout.addWidget(browse_btn)
        self.adjustSize()



    def show_ocr_evaluation_ui(self):
        # Show Grid 2 and Grid 3 for file selection and additional options
        self.grid2_group.setVisible(True)
        self.grid3_group.setVisible(True)
        
        # Hide sample size row for single image processing
        self.sample_size_widget.setVisible(False)
        
        # Reset Grid 2 labels for single image processing
        self.image_label.setText('Select Image:')
        self.image_browse_btn.setText('Browse Image')
        self.image_browse_btn.clicked.disconnect()
        self.image_browse_btn.clicked.connect(self.browse_image)
        
        # Evaluate button
        self.evaluate_button = QPushButton('Evaluate')
        self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #cccccc; color: #666666;')
        self.evaluate_button.clicked.connect(self.evaluate_ocr)
        self.evaluate_button.setEnabled(False)  # Initially disabled
        self.dynamic_layout.addWidget(self.evaluate_button)

        self.adjustSize()

    def show_batch_ocr_evaluation_ui(self):
        # Show Grid 2 and Grid 3 for file selection and additional options
        self.grid2_group.setVisible(True)
        self.grid3_group.setVisible(True)
        
        # Show sample size row for batch processing
        self.sample_size_widget.setVisible(True)
        
        # Update Grid 2 labels for batch processing
        self.image_label.setText('Select Batch Folder:')
        self.image_browse_btn.setText('Browse Batch Folder')
        self.image_browse_btn.clicked.disconnect()
        self.image_browse_btn.clicked.connect(self.browse_batch_folder)
        
        # Info label for valid images
        self.valid_images_info_label = QLabel('')
        self.valid_images_info_label.setStyleSheet('font-size: 12px; color: #333; margin-top: 5px;')
        self.dynamic_layout.addWidget(self.valid_images_info_label)

        # Evaluate button
        self.evaluate_button = QPushButton('Evaluate Batch')
        self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #cccccc; color: #666666;')
        self.evaluate_button.clicked.connect(self.evaluate_batch_ocr)
        self.evaluate_button.setEnabled(False)  # Initially disabled
        self.evaluate_button.setToolTip('Please select batch folder and ground truth map.')
        self.dynamic_layout.addWidget(self.evaluate_button)

        self.adjustSize()

    def validate_file_selections(self):
        """Validate that all required files are selected and enable/disable evaluate button."""
        if self.evaluate_button is None:
            return
            
        # Check if both image and ground truth are selected
        image_selected = self.image_path is not None and os.path.exists(self.image_path)
        gt_selected = self.ground_truth_path is not None and os.path.exists(self.ground_truth_path)
        
        # Additional validation: check if ground truth contains data for the selected image
        gt_valid = False
        if image_selected and gt_selected:
            try:
                import json
                with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                    gt_map = json.load(f)
                image_filename = os.path.basename(self.image_path)
                ground_truth_entries = gt_map.get(image_filename, [])
                gt_valid = len(ground_truth_entries) > 0
            except Exception:
                gt_valid = False
        
        # Enable button only if all validations pass
        all_valid = image_selected and gt_selected and gt_valid
        
        if all_valid:
            self.evaluate_button.setEnabled(True)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #4CAF50; color: white;')
            self.evaluate_button.setToolTip('All required files selected. Ready to evaluate.')
        else:
            self.evaluate_button.setEnabled(False)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #cccccc; color: #666666;')
            
            # Set appropriate tooltip based on what's missing
            if not image_selected:
                self.evaluate_button.setToolTip('Please select an image file.')
            elif not gt_selected:
                self.evaluate_button.setToolTip('Please select a ground truth map file.')
            elif not gt_valid:
                self.evaluate_button.setToolTip('No ground truth data found for the selected image.')
            else:
                self.evaluate_button.setToolTip('Please select all required files.')

    def validate_batch_file_selections(self):
        """Validate that batch folder and ground truth are selected and enable/disable evaluate button."""
        if self.evaluate_button is None:
            return
            
        # Check if both batch folder and ground truth are selected
        batch_selected = self.batch_folder_path is not None and os.path.exists(self.batch_folder_path)
        gt_selected = self.ground_truth_path is not None and os.path.exists(self.ground_truth_path)
        
        # Additional validation: check if ground truth contains data for any images in the batch
        gt_valid = False
        if batch_selected and gt_selected:
            try:
                import json
                with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                    gt_map = json.load(f)
                
                # Get all image files in the batch folder
                image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
                batch_images = [f for f in os.listdir(self.batch_folder_path) 
                              if f.lower().endswith(image_extensions)]
                
                # Check if any images have ground truth data
                for image_file in batch_images:
                    if image_file in gt_map and len(gt_map[image_file]) > 0:
                        gt_valid = True
                        break
                        
            except Exception:
                gt_valid = False
        
        sample_size_valid = False
        if hasattr(self, 'sample_size_spinbox'):
            sample_size = self.sample_size_spinbox.value()
            max_sample = self.sample_size_spinbox.maximum()
            sample_size_valid = 1 <= sample_size <= max_sample
        all_valid = batch_selected and gt_selected and gt_valid and sample_size_valid
        
        if all_valid:
            self.evaluate_button.setEnabled(True)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #4CAF50; color: white;')
            self.evaluate_button.setToolTip('All required files selected. Ready to evaluate batch.')
        else:
            self.evaluate_button.setEnabled(False)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #cccccc; color: #666666;')
            
            # Set appropriate tooltip based on what's missing
            if not batch_selected:
                self.evaluate_button.setToolTip('Please select a batch folder containing images.')
            elif not gt_selected:
                self.evaluate_button.setToolTip('Please select a ground truth map file.')
            elif not gt_valid:
                self.evaluate_button.setToolTip('No ground truth data found for any images in the batch folder.')
            else:
                self.evaluate_button.setToolTip('Please select all required files.')

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image', os.getcwd(), 'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)')
        if file_path:
            self.image_path = file_path
            self.image_path_label.setText(f'Selected: {os.path.basename(file_path)}')
            self.image_path_label.setStyleSheet('font-size: 12px; color: green;')
            # Validate file selections after image selection
            self.validate_file_selections()

    def browse_ground_truth(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Ground Truth Map', os.getcwd(), 'JSON Files (*.json)')
        if file_path:
            self.ground_truth_path = file_path
            self.gt_path_label.setText(f'Selected: {os.path.basename(file_path)}')
            self.gt_path_label.setStyleSheet('font-size: 12px; color: green;')
            # Validate file selections after ground truth selection
            self.validate_file_selections()

    def browse_batch_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Batch Folder', os.getcwd())
        if folder_path:
            self.batch_folder_path = folder_path
            self.image_path_label.setText(f'Selected: {os.path.basename(folder_path)}')
            self.image_path_label.setStyleSheet('font-size: 12px; color: green;')
            
            # Count total images in the folder
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            batch_images = [f for f in os.listdir(self.batch_folder_path) if f.lower().endswith(image_extensions)]
            total_count = len(batch_images)
            
            # Count valid images if ground truth is selected
            valid_count = 0
            if self.ground_truth_path and os.path.exists(self.ground_truth_path):
                try:
                    import json
                    with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                        gt_map = json.load(f)
                    valid_count = sum(1 for f in batch_images if f in gt_map and len(gt_map[f]) > 0)
                except Exception:
                    valid_count = 0
            
            # Update info label
            if self.ground_truth_path and os.path.exists(self.ground_truth_path):
                self.valid_images_info_label.setText(f"Images in folder: {total_count} | Valid for evaluation: {valid_count}")
            else:
                self.valid_images_info_label.setText(f"Images in folder: {total_count} | Select ground truth map to see valid count")
            
            # Enable spinbox and set reasonable defaults
            self.sample_size_spinbox.setEnabled(True)
            if valid_count > 0:
                # If we have valid images, use that count
                self.sample_size_spinbox.setMaximum(max(1, valid_count))
                self.sample_size_spinbox.setValue(min(10, valid_count))
            else:
                # If no ground truth or no valid images, use total count as fallback
                self.sample_size_spinbox.setMaximum(max(1, total_count))
                self.sample_size_spinbox.setValue(min(10, total_count) if total_count > 0 else 1)
            
            # Validate batch file selections after folder selection
            self.validate_batch_file_selections()

    def browse_batch_ground_truth(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Ground Truth Map', os.getcwd(), 'JSON Files (*.json)')
        if file_path:
            self.ground_truth_path = file_path
            self.gt_path_label.setText(f'Selected: {os.path.basename(file_path)}')
            self.gt_path_label.setStyleSheet('font-size: 12px; color: green;')
            
            # Update valid images info and spinbox if batch folder is selected
            if self.batch_folder_path and os.path.exists(self.batch_folder_path):
                try:
                    import json
                    with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                        gt_map = json.load(f)
                    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
                    batch_images = [f for f in os.listdir(self.batch_folder_path) if f.lower().endswith(image_extensions)]
                    total_count = len(batch_images)
                    valid_count = sum(1 for f in batch_images if f in gt_map and len(gt_map[f]) > 0)
                    
                    self.valid_images_info_label.setText(f"Images in folder: {total_count} | Valid for evaluation: {valid_count}")
                    
                    # Update spinbox with valid count
                    self.sample_size_spinbox.setEnabled(True)
                    self.sample_size_spinbox.setMaximum(max(1, valid_count))
                    self.sample_size_spinbox.setValue(min(10, valid_count) if valid_count > 0 else 1)
                except Exception as e:
                    self.valid_images_info_label.setText(f"Error reading ground truth map: {str(e)}")
                    self.sample_size_spinbox.setEnabled(False)
            
            # Validate batch file selections after ground truth selection
            self.validate_batch_file_selections()

    def evaluate_ocr(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Warning', 'Please select an image first.')
            return
        if not self.ground_truth_path:
            QMessageBox.warning(self, 'Warning', 'Please select a ground truth map first.')
            return
        
        # Set processing state
        self.evaluate_button.setEnabled(False)
        self.evaluate_button.setText('Processing...')
        self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #3498db; color: white;')
        
        # Force UI update to show processing state
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            from ocr_evaluation_tool import OCREvaluationTool
            
            # Create evaluator
            levenshtein_threshold = self.levenshtein_threshold_spinbox.value() if hasattr(self, 'levenshtein_threshold_spinbox') else 1
            evaluator = OCREvaluationTool(levenshtein_threshold=levenshtein_threshold)
            import json
            with open(self.ground_truth_path, 'r') as f:
                gt_map = json.load(f)
            image_filename = os.path.basename(self.image_path)
            ground_truth_entries = gt_map.get(image_filename, [])
            
            if not ground_truth_entries:
                QMessageBox.warning(self, 'Warning', f'No ground truth found for image: {image_filename}')
                return
            
            # Extract all text from the ground truth entries
            ground_truth_texts = [entry['text'] for entry in ground_truth_entries if 'text' in entry]
            ground_truth_text = ' '.join(ground_truth_texts)
            
            # Extract bounding boxes for potential region-specific analysis
            ground_truth_bboxes = [entry['bbox'] for entry in ground_truth_entries if 'bbox' in entry]
            
            # Automatically create Evaluation/Run_<timestamp> folder
            eval_dir = os.path.join(os.getcwd(), 'Evaluation')
            run_dir = os.path.join(eval_dir, f'Run_{timestamp}')
            os.makedirs(run_dir, exist_ok=True)

            # Store ground truth entries in evaluator for visualization
            evaluator.ground_truth_entries = ground_truth_entries
            
            # Now call evaluate_image with run_dir - this will:
            # 1. Run OCR on entire image and create OCR_detected_map.json
            # 2. Calculate image quality metrics
            # 3. Perform ground truth comparison
            metrics = evaluator.evaluate_image(self.image_path, ground_truth_text, ground_truth_bboxes, run_dir, ground_truth_entries, self.bbox_tolerance)
            
            # Save PNG and JSON
            png_path = os.path.join(run_dir, f'evaluation_visualization_{timestamp}.png')
            json_path = os.path.join(run_dir, f'evaluation_{timestamp}.json')
            
            # Create visualization with error handling
            try:
                evaluator.create_visualization(self.image_path, png_path)
            except Exception as e:
                # Create a simple fallback visualization
                png_path = self.create_fallback_visualization(run_dir, timestamp, metrics)
            
            evaluator.save_results(json_path)
            
            # Create bounding box visualization if bounding boxes are available
            bbox_viz_path = None
            if ground_truth_bboxes:  # Always create bounding box visualization if available
                bbox_viz_path = os.path.join(run_dir, f'bbox_visualization_{timestamp}.png')
                try:
                    # Store ground truth entries in evaluator for visualization
                    evaluator.ground_truth_entries = ground_truth_entries
                    
                    # Extract ground truth texts
                    ground_truth_texts = [entry.get('text', 'No text') for entry in ground_truth_entries]
                    
                    # Get OCR results for each region
                    ocr_results = []
                    try:
                        import easyocr
                        reader = easyocr.Reader(['en'], gpu=evaluator.use_cuda)
                        import cv2
                        image_gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                        regions = evaluator.extract_bbox_regions(image_gray, ground_truth_bboxes)
                        for region in regions:
                            region_results = reader.readtext(region)
                            if region_results:
                                detected_texts = [text for _, text, _ in region_results]
                                ocr_results.append(" | ".join(detected_texts))
                            else:
                                ocr_results.append("No text detected")
                    except Exception as e:
                        ocr_results = ["OCR error"] * len(ground_truth_bboxes)
                    
                    evaluator.visualize_bbox_regions(image_gray, ground_truth_bboxes, ground_truth_texts, ocr_results, bbox_viz_path)
                    
                except Exception as e:
                    bbox_viz_path = None
            
            # Note: Histogram visualization is already created in evaluate_image method
            # No need to create it again here to avoid duplication
            
            evaluator.clear_gpu_memory()
            
            # Show results window first
            self.show_results_window(png_path, bbox_viz_path)
            
            # Add a small delay to ensure window is visible
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: QMessageBox.information(self, 'Success', 
                f'Evaluation completed successfully!\n\n'
                f'Results saved to:\n{run_dir}\n\n'
                f'Files created:\n'
                f'• OCR_detected_map.json - All detected text regions\n'
                f'• ground_truth_ocr_comparison.json - GT vs OCR comparison\n'
                f'• evaluation_{timestamp}.json - Evaluation metrics\n'
                f'• evaluation_visualization_{timestamp}.png - Metrics visualization\n'
                f'• bbox_visualization_{timestamp}.png - Bounding box visualization\n'
                f'• histogram_analysis_{timestamp}.png - Histogram analysis'))
            
            # Reset button state after successful completion
            self.evaluate_button.setText('Evaluate')
            self.evaluate_button.setEnabled(True)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #4CAF50; color: white;')
            
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, 'Error', f'Failed to evaluate OCR:\n{str(e)}\n\n{tb}')
            
            # Reset button state after error
            self.evaluate_button.setText('Evaluate')
            self.evaluate_button.setEnabled(True)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #4CAF50; color: white;')

    def evaluate_batch_ocr(self):
        """Evaluate OCR performance on a batch of images."""
        if not self.batch_folder_path:
            QMessageBox.warning(self, 'Warning', 'Please select a batch folder first.')
            return
        if not self.ground_truth_path:
            QMessageBox.warning(self, 'Warning', 'Please select a ground truth map first.')
            return
        
        # Set processing state
        self.evaluate_button.setEnabled(False)
        self.evaluate_button.setText('Processing...')
        self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #3498db; color: white;')
        
        # Force UI update to show processing state
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            from ocr_evaluation_tool import OCREvaluationTool
            import json
            
            # Load ground truth map
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                gt_map = json.load(f)
            
            # Get all image files in the batch folder
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            batch_images = [f for f in os.listdir(self.batch_folder_path) 
                          if f.lower().endswith(image_extensions)]
            
            if not batch_images:
                QMessageBox.warning(self, 'Warning', 'No image files found in the selected batch folder.')
                return
            
            # Filter images that have ground truth data
            valid_images = []
            for image_file in batch_images:
                if image_file in gt_map and len(gt_map[image_file]) > 0:
                    valid_images.append(image_file)
            
            if not valid_images:
                QMessageBox.warning(self, 'Warning', 'No images found with ground truth data in the selected batch folder.')
                return
            
            # Randomly select images with true randomization
            import random
            import time
            # Use current time as seed for true randomization
            random.seed(int(time.time() * 1000))  # Use milliseconds for better randomization
            sample_size = self.sample_size_spinbox.value() if hasattr(self, 'sample_size_spinbox') else 10
            sample_size = min(sample_size, len(valid_images))
            selected_images = random.sample(valid_images, sample_size)
            
            # Create evaluator
            levenshtein_threshold = self.levenshtein_threshold_spinbox.value() if hasattr(self, 'levenshtein_threshold_spinbox') else 1
            evaluator = OCREvaluationTool(levenshtein_threshold=levenshtein_threshold)
            
            # Automatically create Evaluation/Batch_Run_<timestamp> folder
            eval_dir = os.path.join(os.getcwd(), 'Evaluation')
            run_dir = os.path.join(eval_dir, f'Batch_Run_{timestamp}')
            os.makedirs(run_dir, exist_ok=True)
            
            # After run_dir is created, before the image processing loop:
            json_dir = os.path.join(run_dir, 'JSON_Files')
            hist_dir = os.path.join(run_dir, 'Histogram')
            map_dir = os.path.join(run_dir, 'MAP')
            eval_dir = os.path.join(run_dir, 'Evaluation')
            for d in [json_dir, hist_dir, map_dir, eval_dir]:
                os.makedirs(d, exist_ok=True)
            
            # In evaluate_batch_ocr, after the other subfolders are created:
            bbox_viz_dir = os.path.join(run_dir, 'BoundingBox_Visualizations')
            os.makedirs(bbox_viz_dir, exist_ok=True)

            # Process each image and collect metrics
            all_metrics = []
            all_ground_truth_entries = []
            processed_images = []
            
            for i, image_file in enumerate(selected_images):
                # Update progress in button text
                progress_text = f'Processing... ({i+1}/{len(selected_images)})'
                self.evaluate_button.setText(progress_text)
                QCoreApplication.processEvents()  # Force UI update
                
                image_path = os.path.join(self.batch_folder_path, image_file)
                ground_truth_entries = gt_map[image_file]
                
                # Extract all text from the ground truth entries
                ground_truth_texts = [entry['text'] for entry in ground_truth_entries if 'text' in entry]
                ground_truth_text = ' '.join(ground_truth_texts)
                
                # Debug text combination
                if i < 3:  # Debug first 3 images
                    print(f"Debug - Image {i+1} ({image_file}) text combination:")
                    print(f"  Ground truth texts: {ground_truth_texts}")
                    print(f"  Combined ground truth text: '{ground_truth_text}'")
                    print(f"  Number of ground truth entries: {len(ground_truth_entries)}")
                
                # Extract bounding boxes for potential region-specific analysis
                ground_truth_bboxes = [entry['bbox'] for entry in ground_truth_entries if 'bbox' in entry]
                
                try:
                    # Store ground truth entries in evaluator for visualization
                    evaluator.ground_truth_entries = ground_truth_entries
                    
                    # Evaluate the image
                    metrics = evaluator.evaluate_image(image_path, ground_truth_text, ground_truth_bboxes, map_dir, ground_truth_entries, tolerance=self.bbox_tolerance)
                    
                    # Add image filename to metrics for tracking
                    metrics['image_filename'] = image_file
                    all_metrics.append(metrics)
                    all_ground_truth_entries.extend(ground_truth_entries)
                    processed_images.append(image_file)
                    
                    # Save individual histogram for this image in the Histogram folder
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    histogram_path = os.path.join(hist_dir, f'histogram_analysis_{image_filename}.png')
                    if hasattr(evaluator, 'create_histogram_visualization'):
                        # Recreate ground_truth_map and ocr_detected_map for this image
                        gt_map_for_hist = {image_file: ground_truth_entries}
                        ocr_map_for_hist = None
                        # Try to load the OCR map from the MAP folder
                        ocr_map_path = os.path.join(map_dir, 'OCR_detected_map.json')
                        if os.path.exists(ocr_map_path):
                            import json
                            with open(ocr_map_path, 'r', encoding='utf-8') as f:
                                ocr_map_for_hist = json.load(f)
                        if ocr_map_for_hist:
                            evaluator.create_histogram_visualization(gt_map_for_hist, ocr_map_for_hist, image_path, histogram_path)

                    # After processing each image (after metrics are added to all_metrics):
                    # Create bounding box visualization for this image
                    if hasattr(evaluator, 'visualize_bbox_regions'):
                        import cv2
                        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        # Get ground truth texts
                        ground_truth_texts = [entry.get('text', 'No text') for entry in ground_truth_entries]
                        
                        # Extract OCR results specifically from each ground truth bounding box region
                        ocr_results = []
                        try:
                            import easyocr
                            reader = easyocr.Reader(['en'], gpu=evaluator.use_cuda)
                            
                            for bbox in ground_truth_bboxes:
                                # Convert percentage coordinates to pixel coordinates
                                x_percent, y_percent, width_percent, height_percent = bbox
                                h, w = image_gray.shape
                                
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
                                    # Extract region and perform OCR
                                    region = image_gray[y:y+height, x:x+width]
                                    region_results = reader.readtext(region)
                                    if region_results:
                                        detected_texts = [text for _, text, _ in region_results]
                                        ocr_results.append(" | ".join(detected_texts))
                                    else:
                                        ocr_results.append("No text detected")
                                else:
                                    ocr_results.append("Invalid region")
                        except Exception as e:
                            print(f"Error extracting OCR results for bounding boxes: {e}")
                            # Fallback: use empty results
                            ocr_results = ["OCR error"] * len(ground_truth_bboxes)
                        
                        # Save visualization
                        bbox_viz_path = os.path.join(bbox_viz_dir, f'bbox_visualization_{os.path.splitext(image_file)[0]}.png')
                        evaluator.visualize_bbox_regions(image_gray, ground_truth_bboxes, ground_truth_texts, ocr_results, bbox_viz_path)

                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not all_metrics:
                QMessageBox.warning(self, 'Warning', 'No images were successfully processed.')
                return
            
            # Calculate aggregate metrics
            aggregate_metrics = self.calculate_aggregate_metrics(all_metrics)
            
            # Save batch results
            # Update all save paths:
            # JSON files
            batch_json_path = os.path.join(json_dir, f'batch_evaluation_{timestamp}.json')
            batch_data_path = os.path.join(json_dir, 'batch_evaluation_data.json')
            # Histogram
            batch_hist_path = os.path.join(hist_dir, 'batch_histogram.png')
            # Evaluation image
            batch_viz_path = os.path.join(eval_dir, f'batch_evaluation_visualization_{timestamp}.png')
            # OCR map (when created, e.g., OCR_detected_map.json)
            # (Assume evaluator saves OCR_detected_map.json, etc. — update path if needed)

            # Update code to use these paths for saving files
            batch_results = {
                'timestamp': timestamp,
                'batch_folder': self.batch_folder_path,
                'ground_truth_map': self.ground_truth_path,
                'total_images': len(batch_images),
                'valid_images': len(valid_images),
                'sampled_images': len(selected_images),
                'processed_images': len(processed_images),
                'selected_image_list': selected_images,
                'processed_image_list': processed_images,
                'aggregate_metrics': aggregate_metrics,
                'individual_metrics': all_metrics
            }
            
            with open(batch_json_path, 'w') as f:
                json.dump(convert_numpy_types(batch_results), f, indent=2)
            with open(batch_data_path, 'w') as f:
                json.dump(convert_numpy_types(all_metrics), f, indent=2)
            
            # Create batch visualization
            try:
                self.create_batch_visualization(aggregate_metrics, all_metrics, batch_viz_path, timestamp)
            except Exception as e:
                batch_viz_path = self.create_fallback_batch_visualization(run_dir, timestamp, aggregate_metrics)
            
            evaluator.clear_gpu_memory()
            
            # Show results window
            self.show_results_window(batch_viz_path)
            
            # Add a small delay to ensure window is visible
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: QMessageBox.information(self, 'Success', 
                f'Batch evaluation completed successfully!\n\n'
                f'Results saved to:\n{run_dir}\n\n'
                f'Files created:\n'
                f'• batch_evaluation_{timestamp}.json - Batch evaluation results\n'
                f'• batch_evaluation_visualization_{timestamp}.png - Batch metrics visualization\n'
                f'• Individual image results in subdirectories\n\n'
                f'Processed {len(processed_images)} out of {len(selected_images)} randomly selected images\n'
                f'(from {len(valid_images)} valid images out of {len(batch_images)} total images)'))
                
            # Create a single comprehensive histogram for all key metrics
            self.create_single_batch_histogram(all_metrics, batch_hist_path)
            
            # Reset button state after successful completion
            self.evaluate_button.setText('Evaluate Batch')
            self.evaluate_button.setEnabled(True)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #4CAF50; color: white;')
                
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, 'Error', f'Failed to evaluate batch OCR:\n{str(e)}\n\n{tb}')
            
            # Reset button state after error
            self.evaluate_button.setText('Evaluate Batch')
            self.evaluate_button.setEnabled(True)
            self.evaluate_button.setStyleSheet('font-size: 16px; padding: 8px 24px; background-color: #4CAF50; color: white;')

    def calculate_aggregate_metrics(self, all_metrics):
        """Calculate aggregate metrics from individual image metrics."""
        if not all_metrics:
            return {}
        
        # Initialize aggregate metrics
        aggregate = {
            'total_images': len(all_metrics),
            'average_ocr_confidence': 0.0,
            'average_text_detection_quality': 0.0,
            'average_word_accuracy': 0.0,
            'average_character_accuracy': 0.0,
            'average_levenshtein_distance': 0.0,
            'average_levenshtein_similarity': 0.0,
            'average_word_precision': 0.0,
            'average_word_recall': 0.0,
            'average_word_f1': 0.0,
            'average_region_detection_rate': 0.0,
            'average_region_accuracy': 0.0,
            'average_quality_score': 0.0,
            'total_words_processed': 0,
            'total_characters_processed': 0,
            'total_regions_processed': 0
        }
        
        # Sum up all metrics with correct key names
        for i, metrics in enumerate(all_metrics):
            # Debug first few metrics to see what values we're getting
            if i < 3:
                print(f"Debug - Aggregate metrics for image {i+1}:")
                print(f"  overall_quality_score: {metrics.get('overall_quality_score', 'NOT FOUND')}")
                print(f"  word_accuracy: {metrics.get('word_accuracy', 'NOT FOUND')}")
                print(f"  ocr_confidence: {metrics.get('ocr_confidence', 'NOT FOUND')}")
                print(f"  character_accuracy: {metrics.get('character_accuracy', 'NOT FOUND')}")
                print(f"  region_detection_accuracy: {metrics.get('region_detection_accuracy', 'NOT FOUND')}")
                print(f"  region_detection_rate: {metrics.get('region_detection_rate', 'NOT FOUND')}")
                print(f"  total_gt_regions: {metrics.get('total_gt_regions', 'NOT FOUND')}")
                print(f"  total_ocr_regions: {metrics.get('total_ocr_regions', 'NOT FOUND')}")
                print(f"  total_gt_words: {metrics.get('total_gt_words', 'NOT FOUND')}")
                print(f"  total_ocr_words: {metrics.get('total_ocr_words', 'NOT FOUND')}")
                print(f"  word_matches: {metrics.get('word_matches', 'NOT FOUND')}")
                print(f"  word_f1: {metrics.get('word_f1', 'NOT FOUND')}")
                print(f"  average_confidence: {metrics.get('average_confidence', 'NOT FOUND')}")
            
            aggregate['average_ocr_confidence'] += metrics.get('ocr_confidence', 0.0)
            aggregate['average_text_detection_quality'] += metrics.get('text_detection_quality', 0.0)
            aggregate['average_word_accuracy'] += metrics.get('word_accuracy', 0.0)
            aggregate['average_character_accuracy'] += metrics.get('character_accuracy', 0.0)
            aggregate['average_levenshtein_distance'] += metrics.get('levenshtein_distance', 0.0)
            aggregate['average_levenshtein_similarity'] += metrics.get('levenshtein_similarity', 0.0)
            aggregate['average_word_precision'] += metrics.get('word_precision', 0.0)
            aggregate['average_word_recall'] += metrics.get('word_recall', 0.0)
            aggregate['average_word_f1'] += metrics.get('word_f1', 0.0)
            aggregate['average_region_detection_rate'] += metrics.get('region_detection_rate', 0.0)
            # Fix: Use the correct key name for region accuracy
            aggregate['average_region_accuracy'] += metrics.get('region_detection_accuracy', 0.0)
            # Fix: Use the correct key name for quality score
            aggregate['average_quality_score'] += metrics.get('overall_quality_score', 0.0)
            
            # Fix: Use the correct keys that are actually returned by evaluate_image
            # For words: use total_gt_words + total_ocr_words (or detected_word_count + ground_truth_word_count as fallback)
            total_words = 0
            if 'total_gt_words' in metrics and 'total_ocr_words' in metrics:
                total_words = metrics.get('total_gt_words', 0) + metrics.get('total_ocr_words', 0)
            elif 'detected_word_count' in metrics and 'ground_truth_word_count' in metrics:
                total_words = metrics.get('detected_word_count', 0) + metrics.get('ground_truth_word_count', 0)
            aggregate['total_words_processed'] += total_words
            
            # For characters: use actual text lengths if available, otherwise estimate from word counts
            total_characters = 0
            if 'average_gt_text_length' in metrics and 'average_ocr_text_length' in metrics:
                # Use actual text lengths from comprehensive metrics
                gt_text_count = metrics.get('gt_text_count', 0)
                ocr_text_count = metrics.get('ocr_text_count', 0)
                avg_gt_length = metrics.get('average_gt_text_length', 0)
                avg_ocr_length = metrics.get('average_ocr_text_length', 0)
                total_characters = int((gt_text_count * avg_gt_length) + (ocr_text_count * avg_ocr_length))
            elif 'total_gt_words' in metrics and 'total_ocr_words' in metrics:
                # Estimate characters from word counts (average 5 chars per word)
                total_characters = (metrics.get('total_gt_words', 0) + metrics.get('total_ocr_words', 0)) * 5
            elif 'detected_word_count' in metrics and 'ground_truth_word_count' in metrics:
                # Estimate characters from word counts (average 5 chars per word)
                total_characters = (metrics.get('detected_word_count', 0) + metrics.get('ground_truth_word_count', 0)) * 5
            aggregate['total_characters_processed'] += total_characters
            
            # For regions: use total_gt_regions + total_ocr_regions
            total_regions = 0
            if 'total_gt_regions' in metrics and 'total_ocr_regions' in metrics:
                total_regions = metrics.get('total_gt_regions', 0) + metrics.get('total_ocr_regions', 0)
            aggregate['total_regions_processed'] += total_regions
        
        # Calculate averages
        num_images = len(all_metrics)
        for key in aggregate:
            if key.startswith('average_') and key != 'total_images':
                aggregate[key] /= num_images
        
        # Debug final aggregate values
        print(f"Debug - Final aggregate metrics:")
        print(f"  average_quality_score: {aggregate.get('average_quality_score', 0.0):.6f}")
        print(f"  average_word_accuracy: {aggregate.get('average_word_accuracy', 0.0):.6f}")
        print(f"  average_ocr_confidence: {aggregate.get('average_ocr_confidence', 0.0):.6f}")
        print(f"  average_character_accuracy: {aggregate.get('average_character_accuracy', 0.0):.6f}")
        print(f"  total_images: {num_images}")
        
        return aggregate

    def create_batch_visualization(self, aggregate_metrics, all_metrics, output_path, timestamp):
        """Create visualization for batch evaluation results with comprehensive histograms."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a comprehensive batch visualization with histograms
        fig = plt.figure(figsize=(20, 16))
        # Lower the suptitle and add extra top margin
        fig.suptitle(f'Batch OCR Evaluation Results (Random Sample of {len(all_metrics)} Images)\n{timestamp}', fontsize=18, fontweight='bold', y=0.96)
        
        # 1. Aggregate metrics summary (top left)
        ax1 = plt.subplot(3, 4, 1)
        metrics_names = ['OCR Confidence', 'Text Quality', 'Word Accuracy', 'Character Accuracy', 'Quality Score']
        metrics_values = [
            aggregate_metrics.get('average_ocr_confidence', 0.0),
            aggregate_metrics.get('average_text_detection_quality', 0.0),
            aggregate_metrics.get('average_word_accuracy', 0.0),
            aggregate_metrics.get('average_character_accuracy', 0.0),
            aggregate_metrics.get('average_quality_score', 0.0)
        ]
        
        # Cap values between 0 and 1 for visualization
        metrics_values = [max(0.0, min(1.0, v)) for v in metrics_values]
        
        bars = ax1.bar(metrics_names, metrics_values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'])
        ax1.set_title('Aggregate Performance Metrics', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
        # 2. Word-level metrics (top middle left)
        ax2 = plt.subplot(3, 4, 2)
        word_metrics = ['Precision', 'Recall', 'F1 Score']
        word_values = [
            aggregate_metrics.get('average_word_precision', 0.0),
            aggregate_metrics.get('average_word_recall', 0.0),
            aggregate_metrics.get('average_word_f1', 0.0)
        ]
        word_values = [max(0.0, min(1.0, v)) for v in word_values]
        
        bars = ax2.bar(word_metrics, word_values, color=['#4CAF50', '#2196F3', '#FF9800'])
        ax2.set_title('Word-Level Performance', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars, word_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Region detection metrics (top middle right)
        ax3 = plt.subplot(3, 4, 3)
        region_metrics = ['Detection Rate', 'Accuracy']
        region_values = [
            aggregate_metrics.get('average_region_detection_rate', 0.0),
            aggregate_metrics.get('average_region_accuracy', 0.0)
        ]
        region_values = [max(0.0, min(1.0, v)) for v in region_values]
        
        bars = ax3.bar(region_metrics, region_values, color=['#4CAF50', '#2196F3'])
        # Fix: Increase padding and move title higher to avoid overlap
        ax3.set_title('Region Detection Performance', fontweight='bold', fontsize=12, pad=30, y=1.10)
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        
        for bar, value in zip(bars, region_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Processing summary (top right)
        ax4 = plt.subplot(3, 4, 4)
        ax4.axis('off')
        summary_text = f"""Batch Processing Summary

Total Images: {aggregate_metrics.get('total_images', 0)}
Valid Images: {len(all_metrics)}
Sampled Images: {len(all_metrics)}
Total Words: {aggregate_metrics.get('total_words_processed', 0)}
Total Characters: {aggregate_metrics.get('total_characters_processed', 0)}
Total Regions: {aggregate_metrics.get('total_regions_processed', 0)}

Average Levenshtein Distance: {aggregate_metrics.get('average_levenshtein_distance', 0.0):.2f}
Average Similarity: {aggregate_metrics.get('average_levenshtein_similarity', 0.0):.3f}

Batch Folder: {os.path.basename(self.batch_folder_path)}
Ground Truth: {os.path.basename(self.ground_truth_path)}"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 5. Quality Score Distribution Histogram (middle left)
        ax5 = plt.subplot(3, 4, 5)
        quality_scores = [m.get('quality_score', 0.0) for m in all_metrics]
        quality_scores = [max(0.0, min(1.0, v)) for v in quality_scores]
        
        counts, _, _ = ax5.hist(quality_scores, bins=8, alpha=0.7, color='#4CAF50', edgecolor='black')
        max_count = max(counts) if len(counts) > 0 else 1
        ax5.set_ylim(0, max_count * 1.15)
        ax5.set_title('Quality Score Distribution', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Quality Score')
        ax5.set_ylabel('Number of Images')
        ax5.axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'Mean: {np.mean(quality_scores):.3f}')
        ax5.legend(fontsize=9)
        
        # 6. OCR Confidence Distribution Histogram (middle middle left)
        ax6 = plt.subplot(3, 4, 6)
        ocr_confidences = [m.get('ocr_confidence', 0.0) for m in all_metrics]
        ocr_confidences = [max(0.0, min(1.0, v)) for v in ocr_confidences]
        
        counts, _, _ = ax6.hist(ocr_confidences, bins=8, alpha=0.7, color='#2196F3', edgecolor='black')
        max_count = max(counts) if len(counts) > 0 else 1
        ax6.set_ylim(0, max_count * 1.15)
        ax6.set_title('OCR Confidence Distribution', fontweight='bold', fontsize=12)
        ax6.set_xlabel('OCR Confidence')
        ax6.set_ylabel('Number of Images')
        ax6.axvline(np.mean(ocr_confidences), color='red', linestyle='--', label=f'Mean: {np.mean(ocr_confidences):.3f}')
        ax6.legend(fontsize=9)
        
        # 7. Word Accuracy Distribution Histogram (middle middle right)
        ax7 = plt.subplot(3, 4, 7)
        word_accuracies = [m.get('word_accuracy', 0.0) for m in all_metrics]
        word_accuracies = [max(0.0, min(1.0, v)) for v in word_accuracies]
        
        counts, _, _ = ax7.hist(word_accuracies, bins=8, alpha=0.7, color='#FF9800', edgecolor='black')
        max_count = max(counts) if len(counts) > 0 else 1
        ax7.set_ylim(0, max_count * 1.15)
        ax7.set_title('Word Accuracy Distribution', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Word Accuracy')
        ax7.set_ylabel('Number of Images')
        ax7.axvline(np.mean(word_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(word_accuracies):.3f}')
        ax7.legend(fontsize=9)
        
        # 8. Character Accuracy Distribution Histogram (middle right)
        ax8 = plt.subplot(3, 4, 8)
        char_accuracies = [m.get('character_accuracy', 0.0) for m in all_metrics]
        char_accuracies = [max(0.0, min(1.0, v)) for v in char_accuracies]
        
        counts, _, _ = ax8.hist(char_accuracies, bins=8, alpha=0.7, color='#9C27B0', edgecolor='black')
        max_count = max(counts) if len(counts) > 0 else 1
        ax8.set_ylim(0, max_count * 1.15)
        ax8.set_title('Character Accuracy Distribution', fontweight='bold', fontsize=12)
        ax8.set_xlabel('Character Accuracy')
        ax8.set_ylabel('Number of Images')
        ax8.axvline(np.mean(char_accuracies), color='red', linestyle='--', label=f'Mean: {np.mean(char_accuracies):.3f}')
        ax8.legend(fontsize=9)
        
        # 9. Word Precision Distribution Histogram (bottom left)
        ax9 = plt.subplot(3, 4, 9)
        word_precisions = [m.get('word_precision', 0.0) for m in all_metrics]
        word_precisions = [max(0.0, min(1.0, v)) for v in word_precisions]
        
        counts, _, _ = ax9.hist(word_precisions, bins=8, alpha=0.7, color='#4CAF50', edgecolor='black')
        max_count = max(counts) if len(counts) > 0 else 1
        ax9.set_ylim(0, max_count * 1.15)
        ax9.set_title('Word Precision Distribution', fontweight='bold', fontsize=12)
        ax9.set_xlabel('Word Precision')
        ax9.set_ylabel('Number of Images')
        ax9.axvline(np.mean(word_precisions), color='red', linestyle='--', label=f'Mean: {np.mean(word_precisions):.3f}')
        ax9.legend(fontsize=9)
        
        # 10. Word Recall Distribution Histogram (bottom middle left)
        ax10 = plt.subplot(3, 4, 10)
        word_recalls = [m.get('word_recall', 0.0) for m in all_metrics]
        word_recalls = [max(0.0, min(1.0, v)) for v in word_recalls]
        
        counts, _, _ = ax10.hist(word_recalls, bins=8, alpha=0.7, color='#2196F3', edgecolor='black')
        max_count = max(counts) if len(counts) > 0 else 1
        ax10.set_ylim(0, max_count * 1.15)
        ax10.set_title('Word Recall Distribution', fontweight='bold', fontsize=12)
        ax10.set_xlabel('Word Recall')
        ax10.set_ylabel('Number of Images')
        ax10.axvline(np.mean(word_recalls), color='red', linestyle='--', label=f'Mean: {np.mean(word_recalls):.3f}')
        ax10.legend(fontsize=9)
        
        # 11. Word F1 Score Distribution Histogram (bottom middle right)
        ax11 = plt.subplot(3, 4, 11)
        word_f1s = [m.get('word_f1', 0.0) for m in all_metrics]
        word_f1s = [max(0.0, min(1.0, v)) for v in word_f1s]
        
        counts, _, _ = ax11.hist(word_f1s, bins=8, alpha=0.7, color='#FF9800', edgecolor='black')
        max_count = max(counts) if len(counts) > 0 else 1
        ax11.set_ylim(0, max_count * 1.15)
        ax11.set_title('Word F1 Score Distribution', fontweight='bold', fontsize=12)
        ax11.set_xlabel('Word F1 Score')
        ax11.set_ylabel('Number of Images')
        ax11.axvline(np.mean(word_f1s), color='red', linestyle='--', label=f'Mean: {np.mean(word_f1s):.3f}')
        ax11.legend(fontsize=9)
        
        # 12. OCR vs Ground Truth Scatter Plot (bottom right)
        ax12 = plt.subplot(3, 4, 12)
        
        # Extract OCR and Ground Truth word counts for scatter plot
        ocr_word_counts = []
        gt_word_counts = []
        image_labels = []
        
        for i, metrics in enumerate(all_metrics):
            # Get OCR word count
            ocr_words = metrics.get('total_ocr_words', 0)
            if ocr_words == 0:
                # Fallback to detected word count
                ocr_words = metrics.get('detected_word_count', 0)
            
            # Get Ground Truth word count
            gt_words = metrics.get('total_gt_words', 0)
            if gt_words == 0:
                # Fallback to ground truth word count
                gt_words = metrics.get('ground_truth_word_count', 0)
            
            if ocr_words > 0 or gt_words > 0:  # Only include points with data
                ocr_word_counts.append(ocr_words)
                gt_word_counts.append(gt_words)
                image_labels.append(f"Img {i+1}")
        
        if ocr_word_counts and gt_word_counts:
            # Create scatter plot
            scatter = ax12.scatter(gt_word_counts, ocr_word_counts, 
                                 c=range(len(ocr_word_counts)), 
                                 cmap='viridis', s=60, alpha=0.7, edgecolors='black')
            
            # Add perfect line (y=x)
            max_val = max(max(gt_word_counts), max(ocr_word_counts)) if gt_word_counts and ocr_word_counts else 10
            ax12.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect Match')
            
            # Add labels for some points
            for i, (gt, ocr, label) in enumerate(zip(gt_word_counts, ocr_word_counts, image_labels)):
                if i < 5:  # Only label first 5 points to avoid clutter
                    ax12.annotate(label, (gt, ocr), xytext=(5, 5), 
                                textcoords='offset points', fontsize=8, alpha=0.8)
            
            ax12.set_xlabel('Ground Truth Words', fontsize=10)
            ax12.set_ylabel('OCR Detected Words', fontsize=10)
            ax12.set_title('OCR vs Ground Truth Word Count', fontweight='bold', fontsize=12)
            ax12.grid(True, alpha=0.3)
            ax12.legend(fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax12, shrink=0.8)
            cbar.set_label('Image Index', fontsize=8)
            
        else:
            ax12.text(0.5, 0.5, 'No word count data available', 
                     ha='center', va='center', transform=ax12.transAxes,
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax12.set_title('OCR vs Ground Truth Word Count', fontweight='bold', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        # At the end of the function, adjust subplot spacing to prevent overlap
        plt.subplots_adjust(top=0.92, hspace=0.6)

    def create_fallback_batch_visualization(self, run_dir, timestamp, aggregate_metrics):
        """Create a simple fallback batch visualization if matplotlib fails."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            
            summary_text = "Batch OCR Evaluation Results\n\n"
            summary_text += f"Timestamp: {timestamp}\n\n"
            summary_text += f"Total Images: {aggregate_metrics.get('total_images', 0)}\n"
            summary_text += f"Average OCR Confidence: {aggregate_metrics.get('average_ocr_confidence', 0.0):.4f}\n"
            summary_text += f"Average Word Accuracy: {aggregate_metrics.get('average_word_accuracy', 0.0):.4f}\n"
            summary_text += f"Average Character Accuracy: {aggregate_metrics.get('average_character_accuracy', 0.0):.4f}\n"
            summary_text += f"Average Quality Score: {aggregate_metrics.get('average_quality_score', 0.0):.4f}\n"
            summary_text += f"Total Words Processed: {aggregate_metrics.get('total_words_processed', 0)}\n"
            summary_text += f"Total Characters Processed: {aggregate_metrics.get('total_characters_processed', 0)}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            fallback_path = os.path.join(run_dir, f'fallback_batch_visualization_{timestamp}.png')
            plt.savefig(fallback_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return fallback_path
            
        except Exception as e:
            print(f"   Debug - Error creating fallback batch visualization: {e}")
            return None

    def show_results_window(self, image_path, bbox_viz_path=None):
        # Check if main visualization file exists
        if not os.path.exists(image_path):
            QMessageBox.warning(self, 'Warning', f'Visualization file not found:\n{image_path}')
            return
        
        if bbox_viz_path and os.path.exists(bbox_viz_path):
            # Show both visualizations
            self.results_window = MultiImageDisplayWindow(image_path, bbox_viz_path)
        else:
            # Show only the main visualization
            self.results_window = ImageDisplayWindow(image_path)
        
        # Make the window more visible
        self.results_window.setWindowState(self.results_window.windowState() | Qt.WindowState.WindowActive)
        self.results_window.raise_()
        self.results_window.activateWindow()
        self.results_window.show()
        
        print(f"   Debug - Results window displayed and activated")
        print(f"   Debug - Window title: {self.results_window.windowTitle()}")
        print(f"   Debug - Window geometry: {self.results_window.geometry()}")

    def create_fallback_visualization(self, run_dir, timestamp, metrics):
        """Create a simple fallback visualization if matplotlib fails."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a simple text-based visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            # Create text summary
            summary_text = "OCR Evaluation Results\n\n"
            summary_text += f"Timestamp: {timestamp}\n\n"
            
            # Add key metrics
            if metrics:
                summary_text += "Key Metrics:\n"
                key_metrics = ['ocr_confidence', 'text_detection_quality', 'word_accuracy', 'levenshtein_distance']
                for metric in key_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, float):
                            summary_text += f"  {metric.replace('_', ' ').title()}: {value:.4f}\n"
                        else:
                            summary_text += f"  {metric.replace('_', ' ').title()}: {value}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Save the fallback visualization
            fallback_path = os.path.join(run_dir, f'fallback_visualization_{timestamp}.png')
            plt.savefig(fallback_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   Debug - Fallback visualization created: {fallback_path}")
            
            return fallback_path
            
        except Exception as e:
            print(f"   Debug - Error creating fallback visualization: {e}")
            # Return None if even fallback fails
            return None

    def browse_export_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select export JSON', os.getcwd(), 'JSON Files (*.json)')
        if file_path:
            try:
                from ground_truth_processor import GroundTruthProcessor
                processor = GroundTruthProcessor(file_path)
                base_dir = os.path.dirname(file_path)
                gt_maps_dir = os.path.join(base_dir, 'Ground_Truth_Maps')
                if not os.path.exists(gt_maps_dir):
                    os.makedirs(gt_maps_dir)
                output_path = os.path.join(gt_maps_dir, 'ground_truth_map.json')
                processor.save_ground_truth_map(output_path)
                
                QMessageBox.information(self, 'Success', f'ground_truth_map.json created at:\n{output_path}')
            except Exception as e:
                tb = traceback.format_exc()
                QMessageBox.critical(self, 'Error', f'Failed to process file:\n{str(e)}\n\n{tb}')

    def create_single_batch_histogram(self, all_metrics, output_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        
        if not all_metrics:
            return
            
        # Create a comprehensive analysis visualization with larger correlation matrix
        fig = plt.figure(figsize=(18, 14))  # Increased figure size
        fig.suptitle('Batch Evaluation: Comprehensive Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Extract metrics for analysis with debugging
        print(f"Debug - Number of metrics entries: {len(all_metrics)}")
        if all_metrics:
            print(f"Debug - First metric entry keys: {list(all_metrics[0].keys())}")
            print(f"Debug - Sample first metric entry: {all_metrics[0]}")
        
        # Try different possible key names for metrics
        quality_scores = []
        ocr_confidences = []
        word_accuracies = []
        char_accuracies = []
        word_precisions = []
        word_recalls = []
        word_f1s = []
        
        for i, m in enumerate(all_metrics):
            # Quality score - try multiple possible keys and fallback to calculated quality
            q_score = m.get('overall_quality_score', 0.0)
            if q_score == 0.0:
                # Fallback: calculate quality score from available metrics
                ocr_conf = m.get('ocr_confidence', 0.0)
                word_acc = m.get('word_accuracy', 0.0)
                char_acc = m.get('character_accuracy', 0.0)
                text_quality = m.get('text_detection_quality', 0.0)
                
                # Calculate a composite quality score from available metrics
                quality_factors = []
                if ocr_conf > 0:
                    quality_factors.append(ocr_conf)
                if word_acc > 0:
                    quality_factors.append(word_acc)
                if char_acc > 0:
                    quality_factors.append(char_acc)
                if text_quality > 0:
                    quality_factors.append(text_quality)
                
                if quality_factors:
                    q_score = np.mean(quality_factors)
                else:
                    q_score = 0.0
            
            quality_scores.append(q_score)
            
            # OCR confidence - use the correct key from evaluate_image
            ocr_conf = m.get('ocr_confidence', 0.0)
            ocr_confidences.append(ocr_conf)
            
            # Word accuracy - use the correct key from evaluate_image
            word_acc = m.get('word_accuracy', 0.0)
            word_accuracies.append(word_acc)
            
            # Character accuracy - use the correct key from evaluate_image
            char_acc = m.get('character_accuracy', 0.0)
            char_accuracies.append(char_acc)
            
            # Word precision - use the correct key from evaluate_image
            word_prec = m.get('word_precision', 0.0)
            word_precisions.append(word_prec)
            
            # Word recall - use the correct key from evaluate_image
            word_rec = m.get('word_recall', 0.0)
            word_recalls.append(word_rec)
            
            # Word F1 - use the correct key from evaluate_image
            word_f1 = m.get('word_f1', 0.0)
            word_f1s.append(word_f1)
            
            # Debug first few entries
            if i < 3:
                print(f"Debug - Image {i+1} metrics:")
                print(f"  overall_quality_score: {q_score}")
                print(f"  ocr_confidence: {ocr_conf}")
                print(f"  word_accuracy: {word_acc}")
                print(f"  character_accuracy: {char_acc}")
                print(f"  word_precision: {word_prec}")
                print(f"  word_recall: {word_rec}")
                print(f"  word_f1: {word_f1}")
                print(f"  text_detection_quality: {m.get('text_detection_quality', 0.0)}")
        
        print(f"Debug - Extracted values summary:")
        print(f"  Quality scores: {quality_scores[:3]}... (mean: {np.mean(quality_scores):.3f})")
        print(f"  OCR confidences: {ocr_confidences[:3]}... (mean: {np.mean(ocr_confidences):.3f})")
        print(f"  Word accuracies: {word_accuracies[:3]}... (mean: {np.mean(word_accuracies):.3f})")
        
        # Debug match-related metrics
        if all_metrics:
            print(f"Debug - Match-related metrics in first entry:")
            match_keys = ['gt_ocr_matching_regions', 'gt_ocr_gt_only_regions', 'gt_ocr_ocr_only_regions', 
                         'total_gt_regions', 'total_ocr_regions', 'gt_ocr_match_rate']
            for key in match_keys:
                value = all_metrics[0].get(key, 'NOT FOUND')
                print(f"  {key}: {value}")
            
            # Check for any keys containing 'match' or 'region'
            match_related_keys = [k for k in all_metrics[0].keys() if 'match' in k.lower() or 'region' in k.lower()]
            if match_related_keys:
                print(f"Debug - All match/region related keys: {match_related_keys}")
            else:
                print(f"Debug - No match/region related keys found")
        
        # Cap most values between 0 and 1, but preserve negative word accuracies for correlation analysis
        quality_scores = [max(0.0, min(1.0, float(v))) for v in quality_scores]
        ocr_confidences = [max(0.0, min(1.0, float(v))) for v in ocr_confidences]
        # Don't cap word accuracies - preserve negative values for correlation analysis
        word_accuracies = [float(v) for v in word_accuracies]
        char_accuracies = [max(0.0, min(1.0, float(v))) for v in char_accuracies]
        word_precisions = [max(0.0, min(1.0, float(v))) for v in word_precisions]
        word_recalls = [max(0.0, min(1.0, float(v))) for v in word_recalls]
        word_f1s = [max(0.0, min(1.0, float(v))) for v in word_f1s]
        
        # Debug the extracted values
        print(f"Debug - Performance distribution values:")
        print(f"  Quality scores (capped): {quality_scores}")
        print(f"  OCR confidences (capped): {ocr_confidences}")
        print(f"  Word accuracies (raw): {word_accuracies}")
        print(f"  Character accuracies (capped): {char_accuracies}")
        print(f"  Word precisions (capped): {word_precisions}")
        print(f"  Word recalls (capped): {word_recalls}")
        print(f"  Word F1s (capped): {word_f1s}")
        
        # 1. Performance Heatmap (top left)
        ax1 = plt.subplot(3, 3, 1)
        # For heatmap, cap word accuracies to 0-1 range to avoid visualization issues
        word_accuracies_heatmap = [max(0.0, min(1.0, v)) for v in word_accuracies]
        metrics_matrix = np.array([quality_scores, ocr_confidences, word_accuracies_heatmap, 
                                  char_accuracies, word_precisions, word_recalls, word_f1s])
        metric_names = ['Quality\nScore', 'OCR\nConfidence', 'Word\nAccuracy', 
                       'Char\nAccuracy', 'Word\nPrecision', 'Word\nRecall', 'Word\nF1']
        
        im = ax1.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(all_metrics)))
        ax1.set_xticklabels([f'Img{i+1}' for i in range(len(all_metrics))], rotation=45, fontsize=8)
        ax1.set_yticks(range(len(metric_names)))
        ax1.set_yticklabels(metric_names, fontsize=9)
        ax1.set_title('Performance Heatmap\n(Green=Good, Red=Poor)', fontweight='bold', fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Performance Score', fontsize=9)
        
        # 2. Metric Correlation Matrix (top middle) - Made much larger with better spacing
        ax2 = plt.subplot(3, 3, 2)
        correlation_data = np.array([quality_scores, ocr_confidences, word_accuracies, 
                                   char_accuracies, word_precisions, word_recalls, word_f1s])
        
        # Calculate correlation with improved error handling to prevent warnings
        try:
            # Check for valid data before correlation calculation
            valid_data = []
            for metric_data in correlation_data:
                # Check if data has variation and is not all zeros
                if (len(set(metric_data)) > 1 and 
                    np.std(metric_data) > 1e-10 and 
                    not np.allclose(metric_data, 0.0)):
                    valid_data.append(metric_data)
                else:
                    # If no variation or all zeros, use zeros instead of random values to avoid misleading correlations
                    valid_data.append(np.zeros(len(metric_data)))
            
            if len(valid_data) >= 2:
                # Suppress warnings during correlation calculation
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    correlation_matrix = np.corrcoef(valid_data)
                
                # Handle NaN and infinite values by replacing with 0
                correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure diagonal elements are 1.0
                np.fill_diagonal(correlation_matrix, 1.0)
            else:
                # Fallback: create identity matrix if insufficient valid data
                correlation_matrix = np.eye(len(correlation_data))
        except Exception:
            # Fallback: create identity matrix if correlation calculation fails
            correlation_matrix = np.eye(len(correlation_data))
        
        im2 = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels([name.replace('\n', ' ') for name in metric_names], rotation=45, fontsize=10)
        ax2.set_yticks(range(len(metric_names)))
        ax2.set_yticklabels([name.replace('\n', ' ') for name in metric_names], fontsize=10)
        ax2.set_title('Metric Correlations\n(Red=Positive, Blue=Negative)', fontweight='bold', fontsize=13, pad=20)
        
        # Add correlation values with no margins/padding around the numbers
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                corr_value = correlation_matrix[i, j]
                if i == j:  # Diagonal elements
                    text = ax2.text(j, i, '1.00',
                                   ha="center", va="center", color="black", fontsize=11, fontweight='bold')
                else:
                    # Format correlation value
                    if abs(corr_value) < 0.01:
                        text_str = '0.00'
                    else:
                        text_str = f'{corr_value:.2f}'
                    
                    # Choose text color based on background
                    if abs(corr_value) > 0.5:
                        text_color = "white"
                    else:
                        text_color = "black"
                    
                    text = ax2.text(j, i, text_str,
                                   ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
        
        # 3. Performance Distribution Box Plot (top right)
        ax3 = plt.subplot(3, 3, 3)
        # For box plot, cap word accuracies to 0-1 range to avoid visualization issues
        word_accuracies_box = [max(0.0, min(1.0, v)) for v in word_accuracies]
        all_metrics_data = [quality_scores, ocr_confidences, word_accuracies_box, 
                           char_accuracies, word_precisions, word_recalls, word_f1s]
        box_labels = ['Quality', 'OCR Conf', 'Word Acc', 'Char Acc', 'Word Prec', 'Word Rec', 'Word F1']
        
        bp = ax3.boxplot(all_metrics_data, tick_labels=box_labels, patch_artist=True)
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#8BC34A']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_title('Performance Distribution\n(Box Plot)', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Trend Analysis (middle left)
        ax4 = plt.subplot(3, 3, 4)
        image_indices = range(1, len(all_metrics) + 1)
        
        # Plot trend lines for key metrics
        ax4.plot(image_indices, quality_scores, 'o-', label='Quality Score', color='#4CAF50', linewidth=2, markersize=6)
        ax4.plot(image_indices, word_accuracies, 's-', label='Word Accuracy', color='#FF9800', linewidth=2, markersize=6)
        ax4.plot(image_indices, ocr_confidences, '^-', label='OCR Confidence', color='#2196F3', linewidth=2, markersize=6)
        
        ax4.set_title('Performance Trends\nAcross Images', fontweight='bold', fontsize=11)
        ax4.set_xlabel('Image Index')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Clustering (middle middle)
        ax5 = plt.subplot(3, 3, 5)
        
        # Create 2D scatter plot of Quality Score vs Word Accuracy
        scatter = ax5.scatter(quality_scores, word_accuracies, c=ocr_confidences, 
                             cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        # Add image labels for first few points
        for i, (q, w) in enumerate(zip(quality_scores, word_accuracies)):
            if i < 5:  # Label first 5 points
                ax5.annotate(f'Img{i+1}', (q, w), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, fontweight='bold')
        
        ax5.set_title('Quality vs Word Accuracy\n(Color=OCR Confidence)', fontweight='bold', fontsize=11)
        ax5.set_xlabel('Quality Score')
        ax5.set_ylabel('Word Accuracy')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter, ax=ax5, shrink=0.8)
        cbar2.set_label('OCR Confidence', fontsize=9)
        
        # 6. Performance Summary Statistics (middle right) - Reduced font size to prevent overlap
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        # Calculate summary statistics
        # For summary stats, cap word accuracies to 0-1 range to avoid misleading statistics
        word_accuracies_stats = [max(0.0, min(1.0, v)) for v in word_accuracies]
        summary_stats = {
            'Quality Score': {'mean': np.mean(quality_scores), 'std': np.std(quality_scores)},
            'OCR Confidence': {'mean': np.mean(ocr_confidences), 'std': np.std(ocr_confidences)},
            'Word Accuracy': {'mean': np.mean(word_accuracies_stats), 'std': np.std(word_accuracies_stats)},
            'Character Accuracy': {'mean': np.mean(char_accuracies), 'std': np.std(char_accuracies)},
            'Word Precision': {'mean': np.mean(word_precisions), 'std': np.std(word_precisions)},
            'Word Recall': {'mean': np.mean(word_recalls), 'std': np.std(word_recalls)},
            'Word F1': {'mean': np.mean(word_f1s), 'std': np.std(word_f1s)}
        }
        
        # Create summary text with shorter format
        summary_text = "Performance Summary\n\n"
        summary_text += f"Total Images: {len(all_metrics)}\n\n"
        
        for metric, stats in summary_stats.items():
            # Shorten metric names to save space
            short_name = metric.replace('Word ', 'W.').replace('Character ', 'Char.').replace('OCR ', 'OCR ')
            summary_text += f"{short_name}:\n"
            summary_text += f"  μ: {stats['mean']:.3f} σ: {stats['std']:.3f}\n"
        
        # Add performance categories
        high_perf = sum(1 for q in quality_scores if q >= 0.8)
        med_perf = sum(1 for q in quality_scores if 0.5 <= q < 0.8)
        low_perf = sum(1 for q in quality_scores if q < 0.5)
        
        # Debug performance categories
        print(f"Debug - Performance categories:")
        print(f"  High performance (≥0.8): {high_perf}")
        print(f"  Medium performance (0.5-0.8): {med_perf}")
        print(f"  Low performance (<0.5): {low_perf}")
        print(f"  Total images: {len(quality_scores)}")
        
        summary_text += f"\nCategories:\n"
        summary_text += f"  High (≥0.8): {high_perf}\n"
        summary_text += f"  Medium (0.5-0.8): {med_perf}\n"
        summary_text += f"  Low (<0.5): {low_perf}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # 7. Performance Histograms (bottom left)
        ax7 = plt.subplot(3, 3, 7)
        
        # Create side-by-side histograms for key metrics
        # Cap word accuracies to 0-1 range for histogram
        word_accuracies_hist = [max(0.0, min(1.0, v)) for v in word_accuracies]
        bins = np.linspace(0, 1, 10)
        ax7.hist(quality_scores, bins=bins, alpha=0.7, label='Quality Score', color='#4CAF50', edgecolor='black')
        ax7.hist(word_accuracies_hist, bins=bins, alpha=0.7, label='Word Accuracy', color='#FF9800', edgecolor='black')
        ax7.hist(ocr_confidences, bins=bins, alpha=0.7, label='OCR Confidence', color='#2196F3', edgecolor='black')
        
        ax7.set_title('Key Metrics Distribution', fontweight='bold', fontsize=11)
        ax7.set_xlabel('Score')
        ax7.set_ylabel('Number of Images')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. Performance Radar Chart (bottom middle)
        ax8 = plt.subplot(3, 3, 8, projection='polar')
        
        # Calculate average values for radar chart
        # Cap word accuracies to 0-1 range for radar chart
        word_accuracies_radar = [max(0.0, min(1.0, v)) for v in word_accuracies]
        avg_metrics = [
            np.mean(quality_scores),
            np.mean(ocr_confidences),
            np.mean(word_accuracies_radar),
            np.mean(char_accuracies),
            np.mean(word_precisions),
            np.mean(word_recalls),
            np.mean(word_f1s)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(avg_metrics), endpoint=False).tolist()
        avg_metrics += avg_metrics[:1]  # Close the polygon
        angles += angles[:1]
        
        ax8.plot(angles, avg_metrics, 'o-', linewidth=2, color='#4CAF50')
        ax8.fill(angles, avg_metrics, alpha=0.25, color='#4CAF50')
        
        # Set labels
        radar_labels = ['Quality', 'OCR Conf', 'Word Acc', 'Char Acc', 'Word Prec', 'Word Rec', 'Word F1']
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(radar_labels, fontsize=8)
        ax8.set_ylim(0, 1)
        ax8.set_title('Average Performance\nRadar Chart', fontweight='bold', fontsize=11, pad=20)
        
        # 9. Performance Insights (bottom right) - Reduced font size and condensed text
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Generate insights with condensed format
        insights = []
        
        # Debug quality scores
        print(f"Debug - Quality scores for insights: {quality_scores}")
        print(f"Debug - Quality scores length: {len(quality_scores)}")
        print(f"Debug - Quality scores max: {np.max(quality_scores) if len(quality_scores) > 0 else 'N/A'}")
        print(f"Debug - Quality scores min: {np.min(quality_scores) if len(quality_scores) > 0 else 'N/A'}")
        
        # Best performing image with error handling
        if len(quality_scores) > 0 and np.max(quality_scores) > 0:
            best_idx = np.argmax(quality_scores)
            insights.append(f"Best: Img{best_idx+1}")
            insights.append(f"  Q: {quality_scores[best_idx]:.3f} W: {word_accuracies[best_idx]:.3f}")
        else:
            insights.append("Best: No valid data")
            insights.append("  Q: N/A W: N/A")
        
        # Worst performing image with error handling
        if len(quality_scores) > 0 and np.min(quality_scores) < np.max(quality_scores):
            worst_idx = np.argmin(quality_scores)
            insights.append(f"\nWorst: Img{worst_idx+1}")
            insights.append(f"  Q: {quality_scores[worst_idx]:.3f} W: {word_accuracies[worst_idx]:.3f}")
        else:
            insights.append(f"\nWorst: No variation")
            insights.append(f"  Q: {quality_scores[0] if len(quality_scores) > 0 else 'N/A'} W: {word_accuracies[0] if len(word_accuracies) > 0 else 'N/A'}")
        
        # Match analysis - check if any matches are being found
        total_matches = 0
        total_gt_regions = 0
        total_ocr_regions = 0
        
        for m in all_metrics:
            # Check for match-related metrics
            matches = m.get('gt_ocr_matching_regions', 0)
            gt_only = m.get('gt_ocr_gt_only_regions', 0)
            ocr_only = m.get('gt_ocr_ocr_only_regions', 0)
            gt_total = m.get('total_gt_regions', 0)
            ocr_total = m.get('total_ocr_regions', 0)
            
            total_matches += matches
            total_gt_regions += gt_total
            total_ocr_regions += ocr_total
        
        insights.append(f"\nMatch Analysis:")
        insights.append(f"  Total Matches: {total_matches}")
        insights.append(f"  GT Regions: {total_gt_regions}")
        insights.append(f"  OCR Regions: {total_ocr_regions}")
        
        if total_gt_regions > 0:
            match_rate = total_matches / total_gt_regions
            insights.append(f"  Match Rate: {match_rate:.3f}")
        else:
            insights.append(f"  Match Rate: N/A")
        
        # Consistency analysis with better debugging
        quality_std = np.std(quality_scores)
        print(f"Debug - Consistency analysis:")
        print(f"  Quality scores: {quality_scores}")
        print(f"  Standard deviation: {quality_std}")
        print(f"  Number of unique values: {len(set(quality_scores))}")
        print(f"  Min: {np.min(quality_scores) if len(quality_scores) > 0 else 'N/A'}")
        print(f"  Max: {np.max(quality_scores) if len(quality_scores) > 0 else 'N/A'}")
        
        if len(quality_scores) == 0:
            consistency = "No Data"
        elif len(set(quality_scores)) == 1:
            consistency = "Identical Values"
        elif quality_std < 0.1:
            consistency = "Very Consistent"
        elif quality_std < 0.2:
            consistency = "Moderately Consistent"
        else:
            consistency = "Highly Variable"
        
        insights.append(f"\nConsistency: {consistency}")
        insights.append(f"Std Dev: {quality_std:.3f}")
        insights.append(f"Unique Values: {len(set(quality_scores))}")
        
        # Correlation insights with detailed debugging
        print(f"Debug - Correlation analysis:")
        print(f"  Quality scores: {quality_scores}")
        print(f"  Word accuracies: {word_accuracies}")
        print(f"  Quality unique values: {len(set(quality_scores))}")
        print(f"  Word accuracy unique values: {len(set(word_accuracies))}")
        print(f"  Quality std dev: {np.std(quality_scores):.6f}")
        print(f"  Word accuracy std dev: {np.std(word_accuracies):.6f}")
        print(f"  Quality min/max: {np.min(quality_scores):.6f}/{np.max(quality_scores):.6f}")
        print(f"  Word accuracy min/max: {np.min(word_accuracies):.6f}/{np.max(word_accuracies):.6f}")
        
        try:
            # Check if data has sufficient variation for correlation
            quality_has_variation = len(set(quality_scores)) > 1 and np.std(quality_scores) > 1e-10
            word_has_variation = len(set(word_accuracies)) > 1 and np.std(word_accuracies) > 1e-10
            
            print(f"  Quality has variation: {quality_has_variation}")
            print(f"  Word accuracy has variation: {word_has_variation}")
            
            if quality_has_variation and word_has_variation:
                # Calculate correlation with better error handling
                quality_word_corr = np.corrcoef(quality_scores, word_accuracies)[0, 1]
                print(f"  Raw correlation: {quality_word_corr}")
                
                # Handle NaN and infinite values
                if np.isnan(quality_word_corr) or np.isinf(quality_word_corr):
                    quality_word_corr = 0.0
                    print(f"  Correlation was NaN/Inf, set to 0.0")
                else:
                    print(f"  Valid correlation calculated: {quality_word_corr:.6f}")
            else:
                quality_word_corr = 0.0
                if not quality_has_variation:
                    print(f"  No variation in quality scores")
                if not word_has_variation:
                    print(f"  No variation in word accuracies")
                print(f"  Correlation set to 0.0 due to lack of variation")
        except Exception as e:
            quality_word_corr = 0.0
            print(f"  Exception in correlation calculation: {e}")
        
        insights.append(f"\nQ-W Corr: {quality_word_corr:.3f}")
        
        # More detailed correlation interpretation with better logic
        if len(quality_scores) == 0 or len(word_accuracies) == 0:
            insights.append("No data available")
        elif len(set(quality_scores)) == 1 and len(set(word_accuracies)) == 1:
            insights.append("No variation in either metric")
        elif len(set(quality_scores)) == 1:
            insights.append("No variation in quality scores")
        elif len(set(word_accuracies)) == 1:
            insights.append("No variation in word accuracies")
        elif abs(quality_word_corr) < 0.01:
            insights.append("No correlation (values are independent)")
        elif quality_word_corr > 0.7:
            insights.append("Strong positive correlation")
        elif quality_word_corr > 0.3:
            insights.append("Moderate positive correlation")
        elif quality_word_corr < -0.3:
            insights.append("Negative correlation")
        else:
            insights.append("Weak correlation")
        
        insights_text = "Performance Insights\n\n" + "\n".join(insights)
        
        ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Use system theme if available
    try:
        import platform
        if platform.system() == 'Windows':
            import ctypes
            try:
                # Enable dark mode for Windows 10/11 if user has it enabled
                ctypes.windll.dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long
                # 20 = DWMWA_USE_IMMERSIVE_DARK_MODE (Windows 10 1809+)
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(app.winId()) if hasattr(app, 'winId') else 0
                value = 1  # 1 = dark, 0 = light
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(ctypes.c_int(value)), 4)
            except Exception:
                pass
        # For all platforms, use the system style
        app.setStyle('Fusion')
        # Optionally, use the system palette
        from PySide6.QtGui import QPalette, QColor
        import os
        if hasattr(QApplication, 'setPalette'):
            app.setPalette(app.style().standardPalette())
    except Exception:
        pass
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 