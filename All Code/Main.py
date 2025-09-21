import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import os
from pathlib import Path
import threading
from PIL import Image, ImageTk
import json
from datetime import datetime

# Import the UI class (assuming it's in a file called ImageTranslatorUI.py)
from ImageTranslatorUI import ImageTranslatorUI

class ImageTranslatorApp:
    """Main application controller that handles business logic"""
    
    def __init__(self, root):
        self.root = root
        
        # Variables
        self.selected_images = []
        self.output_directory = tk.StringVar()
        self.processing = False
        self.translation_source = tk.StringVar(value="asomtavruli")
        
        # Settings
        self.settings_file = "translator_settings.json"
        self.load_settings()
        
        # Create UI first
        self.ui = ImageTranslatorUI(root, self)
        self.ui.log_message("UI created successfully")
        
        # Initialize OCR processor after UI is created
        self.ocr_processor = None
        self.ui.log_message("About to initialize OCR processor...")
        
        try:
            self.initialize_ocr()
            self.ui.log_message("OCR initialization attempt completed")
        except Exception as e:
            self.ui.log_message(f"Exception during OCR initialization: {str(e)}")
            import traceback
            self.ui.log_message(f"OCR init traceback: {traceback.format_exc()}")
        
        # Check final status
        if self.ocr_processor is None:
            self.ui.log_message("WARNING: OCR processor is None - will run in fallback mode")
        else:
            self.ui.log_message("OCR processor successfully initialized")
        
        # Log startup message
        self.ui.log_message("Application started")
        
    def initialize_ocr(self):
        """Initialize the OCR processor with default paths. 
        You'll need to update these paths to match your setup."""
        
        self.ui.log_message("=== OCR INITIALIZATION DEBUG ===")
        self.ui.log_message("Step 1: initialize_ocr() method called")
        
        try:
            # Update these paths to match your actual file locations  
            model_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Asomtavruli Data/Neural Networks/best_dynamic_model_try10_97.60.pth"
            data_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Asomtavruli Data/Sorted"
            font_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Asomtavruli Data/NotoSansGeorgian-VariableFont_wdth,wght.ttf"
            
            self.ui.log_message("Step 2: About to check import of AsomtavruliOCR")
            
            # Test the import first
            try:
                from Asomtavruli_Class import AsomtavruliOCR
                self.ui.log_message("Step 3: AsomtavruliOCR import successful")
            except ImportError as ie:
                self.ui.log_message(f"Step 3 FAILED: Cannot import AsomtavruliOCR: {ie}")
                self.ocr_processor = None
                return
            except Exception as e:
                self.ui.log_message(f"Step 3 FAILED: Import error: {e}")
                self.ocr_processor = None
                return
            
            self.ui.log_message("Step 4: Checking file paths...")
            self.ui.log_message(f"Model path: {model_path}")
            self.ui.log_message(f"Data path: {data_path}")
            self.ui.log_message(f"Font path: {font_path}")
            
            # Check if the essential files exist
            if not os.path.exists(model_path):
                self.ui.log_message(f"Step 5: Model file not found at {model_path}")
                self.ui.log_message("SOLUTION: Update model_path in initialize_ocr() method to your actual model file")
                self.ocr_processor = None
                return
                
            if not os.path.exists(data_path):
                self.ui.log_message(f"Step 5: Data path not found at {data_path}")
                self.ui.log_message("SOLUTION: Update data_path in initialize_ocr() method to your actual data folder")
                self.ocr_processor = None
                return
            
            self.ui.log_message("Step 6: All paths valid, creating AsomtavruliOCR instance...")
            
            # Initialize OCR processor
            self.ocr_processor = AsomtavruliOCR(
                model_path=model_path,
                data_path=data_path,
                font_path=font_path if os.path.exists(font_path) else None,
                output_dir=None,  # Will be set dynamically per processing
                fixed_num_classes=39,
                f0=25,
                num_levels=4,
                blocks_per_level=2,
                dropout_rate=0.18,
                image_size=64
            )
            
            self.ui.log_message("Step 7: OCR processor created successfully!")
            
        except ImportError as e:
            self.ui.log_message(f"IMPORT ERROR: {str(e)}")
            self.ui.log_message("Make sure Asomtavruli_Class.py is in the same directory as Main.py")
            self.ocr_processor = None
        except Exception as e:
            self.ui.log_message(f"GENERAL ERROR during OCR init: {str(e)}")
            import traceback
            self.ui.log_message(f"Full traceback: {traceback.format_exc()}")
            self.ocr_processor = None
            
        self.ui.log_message("=== OCR INITIALIZATION COMPLETE ===")
        self.ui.log_message(f"OCR Processor status: {'READY' if self.ocr_processor else 'FAILED'}")
        
    def select_images(self):
        """Handle image selection via file dialog"""
        filetypes = (
            ('Image files', '*.png *.jpg *.jpeg *.gif *.bmp *.tiff'),
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('All files', '*.*')
        )
        
        files = filedialog.askopenfilenames(
            title='Select Images',
            initialdir=os.getcwd(),
            filetypes=filetypes
        )
        
        if files:
            self.selected_images = list(files)
            self.update_images_display()
            self.ui.log_message(f"Selected {len(self.selected_images)} images")
            
    def clear_images(self):
        """Clear the selected images"""
        self.selected_images = []
        self.update_images_display()
        self.ui.log_message("Cleared selected images")
        
    def on_translation_source_change(self):
        """Handle translation source change"""
        source = self.translation_source.get()
        source_name = "Asomtavruli" if source == "asomtavruli" else "Nuskhuri"
        self.ui.log_message(f"Translation source changed to: {source_name}")
        self.save_settings()
        
    def update_images_display(self):
        """Update the images display in UI"""
        count = len(self.selected_images)
        self.ui.update_images_display(self.selected_images, count)
        
    def select_output_directory(self):
        """Handle output directory selection"""
        directory = filedialog.askdirectory(
            title='Select Output Directory',
            initialdir=self.output_directory.get() or os.getcwd()
        )
        
        if directory:
            self.output_directory.set(directory)
            self.save_settings()
            self.ui.log_message(f"Output directory set to: {directory}")
            
    def start_processing(self):
        """Start the image processing"""
        if not self.selected_images:
            self.ui.show_warning("No Images", "Please select images to process.")
            return
            
        if not self.output_directory.get():
            self.ui.show_warning("No Output Directory", "Please select an output directory.")
            return
            
        if self.processing:
            self.ui.show_info("Processing", "Already processing images. Please wait.")
            return
            
        # Start processing in a separate thread to keep UI responsive
        self.processing = True
        self.ui.update_process_button(False, "Processing...")
        
        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()
        
    def process_images_thread(self):
        """Process images in a separate thread"""
        try:
            total_images = len(self.selected_images)
            processed = 0
            failed = 0
            
            self.ui.log_message(f"Starting batch processing of {total_images} images...")
            
            for i, image_path in enumerate(self.selected_images):
                try:
                    # Update progress
                    progress = (i / total_images) * 100
                    self.ui.update_progress(progress)
                    
                    filename = os.path.basename(image_path)
                    self.ui.update_status(f"Processing {i+1}/{total_images}: {filename} ({self.translation_source.get()})...")
                    self.ui.log_message(f"Processing image {i+1}/{total_images}: {filename}")
                    
                    # Process the image
                    success = self.process_single_image(image_path)
                    
                    if success:
                        processed += 1
                        self.ui.log_message(f"✓ Successfully processed: {filename}")
                    else:
                        failed += 1
                        self.ui.log_message(f"✗ Failed to process: {filename}")
                        
                except Exception as e:
                    failed += 1
                    filename = os.path.basename(image_path)
                    self.ui.log_message(f"✗ Exception processing {filename}: {str(e)}")
                    # Continue with next image instead of stopping
                    continue
                    
            # Final update
            self.ui.update_progress(100)
            self.ui.update_status(f"Batch complete! Processed: {processed}, Failed: {failed}")
            self.ui.log_message(f"Batch processing complete: {processed} successful, {failed} failed out of {total_images} total")
            
        except Exception as e:
            self.ui.log_message(f"Critical error during batch processing: {str(e)}")
            self.ui.update_status("Critical error occurred during processing")
            import traceback
            self.ui.log_message(f"Full traceback: {traceback.format_exc()}")
            
        finally:
            self.processing = False
            self.ui.update_process_button(True)
            
    def process_single_image(self, image_path):
        """
        Process a single image using the AsomtavruliOCR class.
        Uses run_on_image to get a single processed result with predictions drawn.
        """
        try:
            if self.ocr_processor is None:
                self.ui.log_message("OCR processor not initialized. Running in fallback mode...")
                
                # Fallback mode: copy image with modified name for testing
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                source_script = self.translation_source.get()
                
                # Create output directory if it doesn't exist
                os.makedirs(self.output_directory.get(), exist_ok=True)
                
                output_filename = f"ocr_result_{name}_{source_script}.png"
                output_path = os.path.join(self.output_directory.get(), output_filename)
                
                # Copy the original image as a placeholder
                import shutil
                # shutil.copy2(image_path, output_path)
                # self.ui.log_message(f"Fallback mode: Copied {filename} to {output_filename}")
                # return True
                
            # Get the selected translation source
            source_script = self.translation_source.get()
            
            # Set the OCR processor's output directory to our selected output directory
            self.ocr_processor.output_dir = self.output_directory.get()
            os.makedirs(self.ocr_processor.output_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            if source_script == "asomtavruli":
                # Process with Asomtavruli OCR - multiple threshold variants
                self.ui.log_message(f"Running Asomtavruli OCR with all thresholds on {filename}...")
                
                # Use run_on_all_thresholds for multiple outputs (9+ variants)
                saved_paths = self.ocr_processor.run_on_all_thresholds(
                    image_path, 
                    show=False  # Set to True if you want to display results
                )
                
                # Log all generated outputs
                self.ui.log_message(f"Generated {len(saved_paths)} threshold variants:")
                for i, output_path in enumerate(saved_paths):
                    output_filename = os.path.basename(output_path)
                    self.ui.log_message(f"  {i+1}. {output_filename}")
                
            elif source_script == "nuskhuri":
                # For now, use the same Asomtavruli processor
                # TODO: Implement or import your Nuskhuri OCR processor
                self.ui.log_message(f"Nuskhuri OCR not yet implemented. Using Asomtavruli processor with all thresholds for {filename}...")
                
                # Use run_on_all_thresholds for multiple outputs
                saved_paths = self.ocr_processor.run_on_all_thresholds(
                    image_path, 
                    show=False
                )
                
                self.ui.log_message(f"Generated {len(saved_paths)} threshold variants:")
                for i, output_path in enumerate(saved_paths):
                    output_filename = os.path.basename(output_path)
                    self.ui.log_message(f"  {i+1}. {output_filename}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(image_path)}: {str(e)}"
            self.ui.log_message(error_msg)
            # Also log the full traceback for debugging
            import traceback
            self.ui.log_message(f"Full error: {traceback.format_exc()}")
            return False
            
    def open_output_folder(self):
        """Open the output folder in the file explorer"""
        if self.output_directory.get() and os.path.exists(self.output_directory.get()):
            if os.name == 'nt':  # Windows
                os.startfile(self.output_directory.get())
            elif os.name == 'posix':  # macOS and Linux
                import sys
                os.system(f'open "{self.output_directory.get()}"' if sys.platform == 'darwin' 
                         else f'xdg-open "{self.output_directory.get()}"')
        else:
            self.ui.show_warning("Invalid Directory", "Output directory not set or doesn't exist.")
            
    def save_settings(self):
        """Save application settings"""
        settings = {
            'output_directory': self.output_directory.get(),
            'translation_source': self.translation_source.get()
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            self.ui.log_message(f"Failed to save settings: {str(e)}")
            
    def load_settings(self):
        """Load application settings"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.output_directory.set(settings.get('output_directory', ''))
                    self.translation_source.set(settings.get('translation_source', 'asomtavruli'))
        except Exception as e:
            pass  # Use defaults if loading fails
            
    def update_ocr_paths(self, model_path, data_path, font_path=None):
        """Update OCR processor paths and reinitialize if needed"""
        try:
            if os.path.exists(model_path) and os.path.exists(data_path):
                self.ocr_processor = AsomtavruliOCR(
                    model_path=model_path,
                    data_path=data_path,
                    font_path=font_path if font_path and os.path.exists(font_path) else None,
                    output_dir=self.output_directory.get() or os.getcwd(),
                    fixed_num_classes=39,
                    f0=25,
                    num_levels=4,
                    blocks_per_level=2,
                    dropout_rate=0.18,
                    image_size=64
                )
                self.ui.log_message("OCR processor paths updated successfully")
                return True
            else:
                self.ui.log_message("Invalid paths provided for OCR processor")
                return False
        except Exception as e:
            self.ui.log_message(f"Failed to update OCR processor: {str(e)}")
            return False


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = ImageTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    main()