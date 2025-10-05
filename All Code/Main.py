import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import os
from pathlib import Path
import threading
from PIL import Image, ImageTk
import json
from datetime import datetime
from Asomtavruli_Class import AsomtavruliOCR
from Nuskhuri_Class import NuskhuriOCR 
# Import the UI class (assuming it's in a file called ImageTranslatorUI.py)
from ImageTranslatorUI import ImageTranslatorUI


#TODO: find and replace Nuskhuri font path when available
#

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
        self.asomtavruli_ocr = None
        self.nuskhuri_ocr = None
        self.ui.log_message("About to initialize OCR processors...")
        
        try:
            self.initialize_ocr()
            self.ui.log_message("OCR initialization attempt completed")
        except Exception as e:
            self.ui.log_message(f"Exception during OCR initialization: {str(e)}")
            import traceback
            self.ui.log_message(f"OCR init traceback: {traceback.format_exc()}")
        
        # Check final status
        if self.asomtavruli_ocr is None and self.nuskhuri_ocr is None:
            self.ui.log_message("WARNING: No OCR processors initialized - will run in fallback mode")
        else:
            status_msgs = []
            if self.asomtavruli_ocr: status_msgs.append("Asomtavruli")
            if self.nuskhuri_ocr: status_msgs.append("Nuskhuri")
            self.ui.log_message(f"OCR processors successfully initialized: {', '.join(status_msgs)}")
        
        # Log startup message
        self.ui.log_message("Application started")
        
    def initialize_ocr(self):
        """Initialize both OCR processors with their respective paths."""
        
        self.ui.log_message("=== OCR INITIALIZATION DEBUG ===")
        self.ui.log_message("Step 1: initialize_ocr() method called")

        
        # Initialize Asomtavruli OCR
        self.ui.log_message("Step 2: Initializing Asomtavruli OCR...")
        try:
            asomtavruli_model_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Asomtavruli Data/Neural Networks/best_dynamic_model_try10_97.60.pth"
            asomtavruli_data_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Asomtavruli Data/Sorted"
            # Define font_path once so it's available for both blocks even if the first fails early
            font_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Asomtavruli Data/NotoSansGeorgian-VariableFont_wdth,wght.ttf"
            font_path_if_exists = font_path if os.path.exists(font_path) else None
            
            if os.path.exists(asomtavruli_model_path) and os.path.exists(asomtavruli_data_path):
                self.asomtavruli_ocr = AsomtavruliOCR(
                    model_path=asomtavruli_model_path,
                    data_path=asomtavruli_data_path,
                    font_path=font_path_if_exists,
                    output_dir=None,
                    fixed_num_classes=39,
                    f0=25,
                    num_levels=4,
                    blocks_per_level=2,
                    dropout_rate=0.18,
                    image_size=64
                )
                self.ui.log_message("✓ Asomtavruli OCR initialized successfully")
            else:
                self.ui.log_message("✗ Asomtavruli files not found:")
                self.ui.log_message(f"  Model: {asomtavruli_model_path} ({'Found' if os.path.exists(asomtavruli_model_path) else 'Not Found'})")
                self.ui.log_message(f"  Data: {asomtavruli_data_path} ({'Found' if os.path.exists(asomtavruli_data_path) else 'Not Found'})")
        except Exception as e:
            self.ui.log_message(f"✗ Asomtavruli OCR initialization failed: {str(e)}")
            self.asomtavruli_ocr = None

        # Initialize Nuskhuri OCR
        self.ui.log_message("Step 3: Initializing Nuskhuri OCR...")
        try:
            nuskhuri_model_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Nuskhuri Data/Neural Networks/Legacy Models_99.17.pth"
            nuskhuri_data_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Nuskhuri Data/Sorted"
            # Define font_path once so it's available for both blocks even if the first fails early
            #TODO: Change to Nuskhuri font when available
            font_path = "/Users/sunnysideup/Documents/Georgian-Script-Translator-Thesis/Asomtavruli Data/NotoSansGeorgian-VariableFont_wdth,wght.ttf"
            font_path_if_exists = font_path if os.path.exists(font_path) else None
            
            if os.path.exists(nuskhuri_model_path) and os.path.exists(nuskhuri_data_path):
                self.nuskhuri_ocr = NuskhuriOCR(
                    model_path=nuskhuri_model_path,
                    data_path=nuskhuri_data_path,
                    font_path=font_path_if_exists,
                    output_dir=None,
                    fixed_num_classes=42,
                    f0=25,
                    num_levels=4,
                    blocks_per_level=2,
                    dropout_rate=0.18,
                    image_size=64
                )
                self.ui.log_message("✓ Nuskhuri OCR initialized successfully")
            else:
                self.ui.log_message("✗ Nuskhuri files not found:")
                self.ui.log_message(f"  Model: {nuskhuri_model_path} ({'Found' if os.path.exists(nuskhuri_model_path) else 'Not Found'})")
                self.ui.log_message(f"  Data: {nuskhuri_data_path} ({'Found' if os.path.exists(nuskhuri_data_path) else 'Not Found'})")
                self.ui.log_message("  Note: Update nuskhuri_model_path and nuskhuri_data_path in initialize_ocr() method")
        except Exception as e:
            self.ui.log_message(f"✗ Nuskhuri OCR initialization failed: {str(e)}")
            self.nuskhuri_ocr = None
            
        self.ui.log_message("=== OCR INITIALIZATION COMPLETE ===")
        
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
        Process a single image using the selected OCR engine.
        Uses run_on_all_thresholds to produce multiple variants.
        """
        try:
            source_script = self.translation_source.get()

            # Pick the OCR implementation based on the current selection
            if source_script == "asomtavruli":
                ocr = self.asomtavruli_ocr
                ocr_name = "Asomtavruli"
            else:
                ocr = self.nuskhuri_ocr
                ocr_name = "Nuskhuri"

            # If the selected OCR isn't initialized, try to fall back to the other one
            if ocr is None:
                fallback_ocr = self.asomtavruli_ocr if source_script != "asomtavruli" else self.nuskhuri_ocr
                fallback_name = "Asomtavruli" if source_script != "asomtavruli" else "Nuskhuri"
                if fallback_ocr is not None:
                    self.ui.log_message(f"{ocr_name} OCR not available. Falling back to {fallback_name}.")
                    ocr = fallback_ocr
                    ocr_name = fallback_name
                else:
                    # True fallback: just copy/emit placeholder so the batch can continue
                    self.ui.log_message("No OCR engines initialized. Running in placeholder fallback mode...")
                    filename = os.path.basename(image_path)
                    name, _ = os.path.splitext(filename)
                    os.makedirs(self.output_directory.get(), exist_ok=True)
                    output_filename = f"ocr_result_{name}_{source_script}.png"
                    output_path = os.path.join(self.output_directory.get(), output_filename)
                    # If you want an actual copy, uncomment these lines:
                    # import shutil
                    # shutil.copy2(image_path, output_path)
                    # self.ui.log_message(f"Fallback: copied original to {output_filename}")
                    return True  # Don't fail the batch

            # Ensure the OCR’s output m.kmdir matches current selection
            ocr.output_dir = self.output_directory.get()
            os.makedirs(ocr.output_dir, exist_ok=True)

            filename = os.path.basename(image_path)
            self.ui.log_message(f"Running {ocr_name} OCR with all thresholds on {filename}...")

            # Run the multi-threshold pipeline
            saved_paths = ocr.run_on_all_thresholds(image_path, show=False)

            self.ui.log_message(f"Generated {len(saved_paths)} threshold variants:")
            for i, out_path in enumerate(saved_paths, start=1):
                self.ui.log_message(f"  {i}. {os.path.basename(out_path)}")

            return True

        except Exception as e:
            error_msg = f"Error processing {os.path.basename(image_path)}: {str(e)}"
            self.ui.log_message(error_msg)
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
        
    
            
    


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = ImageTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    main()