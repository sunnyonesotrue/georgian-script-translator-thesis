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
        
        # Create UI (pass this controller to the UI)
        self.ui = ImageTranslatorUI(root, self)
        
        # Log startup message
        self.ui.log_message("Application started")
        
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
            
            for i, image_path in enumerate(self.selected_images):
                # Update progress
                progress = (i / total_images) * 100
                self.ui.update_progress(progress)
                
                filename = os.path.basename(image_path)
                self.ui.update_status(f"Processing {filename} ({self.translation_source.get()})...")
                
                # Process the image
                success = self.process_single_image(image_path)
                
                if success:
                    processed += 1
                    self.ui.log_message(f"✓ Processed: {filename}")
                else:
                    self.ui.log_message(f"✗ Failed: {filename}")
                    
            # Final update
            self.ui.update_progress(100)
            self.ui.update_status(f"Completed! Processed {processed}/{total_images} images")
            self.ui.log_message(f"Processing complete: {processed}/{total_images} images successful")
            
        except Exception as e:
            self.ui.log_message(f"Error during processing: {str(e)}")
            self.ui.update_status("Error occurred during processing")
            
        finally:
            self.processing = False
            self.ui.update_process_button(True)
            
    def process_single_image(self, image_path):
        """
        Process a single image. Replace this with your actual OCR processing logic.
        
        This is where you'll integrate your Georgian script OCR system.
        """
        try:
            # Get the selected translation source
            source_script = self.translation_source.get()
            
            # Simulate processing time (remove this in actual implementation)
            import time
            time.sleep(0.1)  # Remove this line
            
            # TODO: Replace this with your OCR processing
            # Example integration points:
            
            # 1. Load the image
            # from your_ocr_module import load_image
            # original_image = load_image(image_path)
            
            # 2. Run OCR based on selected script
            # if source_script == "asomtavruli":
            #     from your_ocr_module import asomtavruli_ocr
            #     extracted_text = asomtavruli_ocr(original_image)
            # elif source_script == "nuskhuri":
            #     from your_ocr_module import nuskhuri_ocr
            #     extracted_text = nuskhuri_ocr(original_image)
            
            # 3. Translate the extracted text
            # from your_translation_module import translate_text
            # translated_text = translate_text(extracted_text, target_language="modern_georgian")
            
            # 4. Create new image with translated text
            # from your_image_module import create_translated_image
            # translated_image = create_translated_image(original_image, translated_text)
            
            # 5. Save the translated image
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_{source_script}_translated{ext}"
            output_path = os.path.join(self.output_directory.get(), output_filename)
            
            # Placeholder: Just copy the original image for now
            # Replace this with: translated_image.save(output_path)
            import shutil
            shutil.copy2(image_path, output_path)
            
            return True
            
        except Exception as e:
            self.ui.log_message(f"Error processing {os.path.basename(image_path)}: {str(e)}")
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