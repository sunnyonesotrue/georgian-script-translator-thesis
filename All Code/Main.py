import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import os
from pathlib import Path
import threading
from PIL import Image, ImageTk
import json
from datetime import datetime

class ImageTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image OCR Translator")
        self.root.geometry("800x600")
        self.root.minsize(650, 500)  # Increased minimum size to ensure all elements are visible
        
        # Variables
        self.selected_images = []
        self.output_directory = tk.StringVar()
        self.processing = False
        self.translation_source = tk.StringVar(value="asomtavruli")  # Default to Asomtavruli
        
        # Load settings
        self.settings_file = "translator_settings.json"
        self.load_settings()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsive resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Set row weights to control how space is distributed
        main_frame.rowconfigure(0, weight=0)  # Title - fixed
        main_frame.rowconfigure(1, weight=0)  # Input section - fixed
        main_frame.rowconfigure(2, weight=0)  # Translation source section - fixed
        main_frame.rowconfigure(3, weight=0)  # Output section - fixed  
        main_frame.rowconfigure(4, weight=1)  # Preview section - expandable
        main_frame.rowconfigure(5, weight=0)  # Progress section - fixed
        main_frame.rowconfigure(6, weight=1)  # Log section - expandable
        main_frame.rowconfigure(7, weight=0)  # Buttons - fixed
        
        # Title
        title_label = ttk.Label(main_frame, text="Image OCR Translator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Input Images", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Button(input_frame, text="Select Images", 
                  command=self.select_images).grid(row=0, column=0, padx=(0, 10))
        
        self.images_label = ttk.Label(input_frame, text="No images selected")
        self.images_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Button(input_frame, text="Clear", 
                  command=self.clear_images).grid(row=0, column=2)
        
        # Translation Source Section
        translation_frame = ttk.LabelFrame(main_frame, text="Translation Source", padding="10")
        translation_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(translation_frame, text="Translate from:").grid(row=0, column=0, padx=(0, 15), sticky=tk.W)
        
        asomtavruli_radio = ttk.Radiobutton(translation_frame, text="Asomtavruli (ⴀⴑⴍⴋⴇⴀⴅⴐⴓⴊⴈ)", 
                                          variable=self.translation_source, value="asomtavruli",
                                          command=self.on_translation_source_change)
        asomtavruli_radio.grid(row=0, column=1, padx=(0, 20), sticky=tk.W)
        
        nuskhuri_radio = ttk.Radiobutton(translation_frame, text="Nuskhuri (ⴌⴓⴑⴞⴓⴐⴈ)", 
                                       variable=self.translation_source, value="nuskhuri",
                                       command=self.on_translation_source_change)
        nuskhuri_radio.grid(row=0, column=2, sticky=tk.W)
        
        # Output Section
        output_frame = ttk.LabelFrame(main_frame, text="Output Location", padding="10")
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Button(output_frame, text="Select Output Folder", 
                  command=self.select_output_directory).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Entry(output_frame, textvariable=self.output_directory, 
                 state="readonly").grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Image Preview Section - with minimum height
        preview_frame = ttk.LabelFrame(main_frame, text="Selected Images", padding="10")
        preview_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Listbox for images with minimum height
        self.images_listbox = tk.Listbox(preview_frame, height=4)  # Reduced from 6 to 4 for better space management
        self.images_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.images_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.images_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.status_label = ttk.Label(progress_frame, text="Ready to process images")
        self.status_label.grid(row=1, column=0, sticky=tk.W)
        
        # Log Section - with minimum height
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=6, width=70)  # Reduced from 8 to 6
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=(10, 0))
        
        self.process_button = ttk.Button(button_frame, text="Process Images", 
                                       command=self.start_processing, style="Accent.TButton")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Open Output Folder", 
                  command=self.open_output_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.RIGHT)
        
    def select_images(self):
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
            self.log(f"Selected {len(self.selected_images)} images")
        
    def clear_images(self):
        self.selected_images = []
        self.update_images_display()
        self.log("Cleared selected images")
    
    def on_translation_source_change(self):
        source = self.translation_source.get()
        source_name = "Asomtavruli" if source == "asomtavruli" else "Nuskhuri"
        self.log(f"Translation source changed to: {source_name}")
        
    def update_images_display(self):
        count = len(self.selected_images)
        if count == 0:
            self.images_label.config(text="No images selected")
        else:
            self.images_label.config(text=f"{count} images selected")
            
        # Update listbox
        self.images_listbox.delete(0, tk.END)
        for img_path in self.selected_images:
            filename = os.path.basename(img_path)
            self.images_listbox.insert(tk.END, filename)
    
    def select_output_directory(self):
        directory = filedialog.askdirectory(
            title='Select Output Directory',
            initialdir=self.output_directory.get() or os.getcwd()
        )
        
        if directory:
            self.output_directory.set(directory)
            self.save_settings()
            self.log(f"Output directory set to: {directory}")
    
    def start_processing(self):
        if not self.selected_images:
            messagebox.showwarning("No Images", "Please select images to process.")
            return
            
        if not self.output_directory.get():
            messagebox.showwarning("No Output Directory", "Please select an output directory.")
            return
            
        if self.processing:
            messagebox.showinfo("Processing", "Already processing images. Please wait.")
            return
            
        # Start processing in a separate thread to keep UI responsive
        self.processing = True
        self.process_button.config(state="disabled", text="Processing...")
        
        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()
    
    def process_images_thread(self):
        try:
            total_images = len(self.selected_images)
            processed = 0
            
            for i, image_path in enumerate(self.selected_images):
                # Update progress
                progress = (i / total_images) * 100
                self.progress_var.set(progress)
                
                filename = os.path.basename(image_path)
                self.update_status(f"Processing {filename} ({self.translation_source.get()})...")
                
                # Here's where you integrate your OCR system
                # Replace this with your actual OCR processing
                success = self.process_single_image(image_path)
                
                if success:
                    processed += 1
                    self.log(f"✓ Processed: {filename}")
                else:
                    self.log(f"✗ Failed: {filename}")
                
            # Final update
            self.progress_var.set(100)
            self.update_status(f"Completed! Processed {processed}/{total_images} images")
            self.log(f"Processing complete: {processed}/{total_images} images successful")
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            self.update_status("Error occurred during processing")
            
        finally:
            self.processing = False
            self.process_button.config(state="normal", text="Process Images")
    
    def process_single_image(self, image_path):
        """
        Replace this method with your actual OCR processing logic.
        This is just a placeholder that copies the image to the output directory.
        """
        try:
            # Get the selected translation source
            source_script = self.translation_source.get()
            
            # Simulate processing time (remove this in actual implementation)
            import time
            time.sleep(0.1)  # Remove this line
            
            # TODO: Replace this with your OCR processing
            # 1. Load the image
            # 2. Run OCR to extract text from the specified script (Asomtavruli or Nuskhuri)
            # 3. Translate the text to modern Georgian or target language
            # 4. Create translated image
            # 5. Save to output directory
            
            # Example of how to use the source script selection:
            # if source_script == "asomtavruli":
            #     extracted_text = your_asomtavruli_ocr_function(image_path)
            # elif source_script == "nuskhuri":
            #     extracted_text = your_nuskhuri_ocr_function(image_path)
            
            # Placeholder: Just copy the original image for now
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_{source_script}_translated{ext}"
            output_path = os.path.join(self.output_directory.get(), output_filename)
            
            # Simple copy operation (replace with your OCR logic)
            import shutil
            shutil.copy2(image_path, output_path)
            
            return True
            
        except Exception as e:
            self.log(f"Error processing {os.path.basename(image_path)}: {str(e)}")
            return False
    
    def update_status(self, message):
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
        
        self.root.after(0, update_log)
    
    def open_output_folder(self):
        if self.output_directory.get() and os.path.exists(self.output_directory.get()):
            if os.name == 'nt':  # Windows
                os.startfile(self.output_directory.get())
            elif os.name == 'posix':  # macOS and Linux
                os.system(f'open "{self.output_directory.get()}"' if sys.platform == 'darwin' 
                         else f'xdg-open "{self.output_directory.get()}"')
        else:
            messagebox.showwarning("Invalid Directory", "Output directory not set or doesn't exist.")
    
    def save_settings(self):
        settings = {
            'output_directory': self.output_directory.get(),
            'translation_source': self.translation_source.get()
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            self.log(f"Failed to save settings: {str(e)}")
    
    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.output_directory.set(settings.get('output_directory', ''))
                    # Load translation source setting (defaults to asomtavruli)
                    self.translation_source.set(settings.get('translation_source', 'asomtavruli'))
        except Exception as e:
            pass  # Use defaults if loading fails

def main():
    root = tk.Tk()
    app = ImageTranslatorApp(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    import sys
    main()