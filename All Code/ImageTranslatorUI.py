import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import os
from pathlib import Path
import threading
from PIL import Image, ImageTk
import json
from datetime import datetime

class ImageTranslatorUI:
    """Handles all UI-related functionality"""
    
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.ttk = ttk
        self.setup_window()
        self.setup_ui()
        
    def setup_window(self):
        """Configure the main window"""
        self.root.title("Image OCR Translator")
        self.root.geometry("800x600")
        self.root.minsize(650, 500)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
    def setup_ui(self):
        """Setup the user interface"""
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
        main_frame.rowconfigure(4, weight=0)  # Text generation option - fixed (NEW)
        main_frame.rowconfigure(5, weight=0)  # Preview section - expandable
        main_frame.rowconfigure(6, weight=0)  # Progress section - fixed
        main_frame.rowconfigure(7, weight=1)  # Log section - expandable
        main_frame.rowconfigure(8, weight=0)  # Buttons - fixed
        
        self.create_title_section(main_frame)
        self.create_input_section(main_frame)
        self.create_translation_source_section(main_frame)
        self.create_output_section(main_frame)
        self.create_text_generation_section(main_frame)
        # self.create_text_generation_translate_section(main_frame)
        self.create_preview_section(main_frame)
        self.create_progress_section(main_frame)
        self.create_log_section(main_frame)
        self.create_button_section(main_frame)
        
    def create_title_section(self, parent):
        """Create the title section"""
        title_label = ttk.Label(parent, text="Image OCR Translator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
    def create_input_section(self, parent):
        """Create the input images section"""
        input_frame = ttk.LabelFrame(parent, text="Input Images", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Button(input_frame, text="Select Images", 
                  command=self.controller.select_images).grid(row=0, column=0, padx=(0, 10))
        
        self.images_label = ttk.Label(input_frame, text="No images selected")
        self.images_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Button(input_frame, text="Clear", 
                  command=self.controller.clear_images).grid(row=0, column=2)
                  
    def create_translation_source_section(self, parent):
        """Create the translation source selection section"""
        translation_frame = ttk.LabelFrame(parent, text="Translation Source", padding="10")
        translation_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(translation_frame, text="Translate from:").grid(row=0, column=0, padx=(0, 15), sticky=tk.W)
        
        asomtavruli_radio = ttk.Radiobutton(translation_frame, text="Asomtavruli (ⴀⴑⴍⴋⴇⴀⴅⴐⴓⴊⴈ)", 
                                          variable=self.controller.translation_source, value="asomtavruli",
                                          command=self.controller.on_translation_source_change)
        asomtavruli_radio.grid(row=0, column=1, padx=(0, 20), sticky=tk.W)
        
        nuskhuri_radio = ttk.Radiobutton(translation_frame, text="Nuskhuri (ⴌⴓⴑⴞⴓⴐⴈ)", 
                                       variable=self.controller.translation_source, value="nuskhuri",
                                       command=self.controller.on_translation_source_change)
        nuskhuri_radio.grid(row=0, column=2, sticky=tk.W)
        
    def create_output_section(self, parent):
        """Create the output directory section"""
        output_frame = ttk.LabelFrame(parent, text="Output Location", padding="10")
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Button(output_frame, text="Select Output Folder", 
                  command=self.controller.select_output_directory).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Entry(output_frame, textvariable=self.controller.output_directory, 
                 state="readonly").grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
    
    def create_text_generation_section(self, parent):
        """Create the text file generation option section (NEW)"""
        text_gen_frame = ttk.LabelFrame(parent, text="Output Options", padding="10")
        text_gen_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.text_gen_checkbox = ttk.Checkbutton(
            text_gen_frame, 
            text="Generate text files with OCR results (spatially preserved)",
            variable=self.controller.generate_text_files,
            command=self.controller.save_settings
        )
        self.text_gen_translate_checkbox = ttk.Checkbutton(
            text_gen_frame, 
            text="Translate OCR results to Modern Georgian",
            variable=self.controller.translate_to_modern,
            command=self.controller.save_settings
        )
        self.text_gen_checkbox.grid(row=0, column=0, sticky=tk.W)
        self.text_gen_translate_checkbox.grid(row=1, column=0, sticky=tk.W)
        
    # def create_text_generation_translate_section(self, parent):
    #     """Create the text file generation option section (NEW)"""
    #     text_gen_frame = ttk.LabelFrame(parent, text="Output Options", padding="10")
    #     text_gen_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # self.text_gen_checkbox = ttk.Checkbutton(
        #     text_gen_frame, 
        #     text="Translate OCR results to Modern Georgian",
        #     variable=self.controller.translate_to_modern,
        #     command=self.controller.save_settings
        # )
    #     self.text_gen_checkbox.grid(row=0, column=0, sticky=tk.W)
                 
    def create_preview_section(self, parent):
        """Create the image preview section"""
        preview_frame = ttk.LabelFrame(parent, text="Selected Images", padding="10")
        preview_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Listbox for images with minimum height
        self.images_listbox = tk.Listbox(preview_frame, height=4)
        self.images_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.images_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.images_listbox.configure(yscrollcommand=scrollbar.set)
        
    def create_progress_section(self, parent):
        """Create the progress section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.status_label = ttk.Label(progress_frame, text="Ready to process images")
        self.status_label.grid(row=1, column=0, sticky=tk.W)
        
    def create_log_section(self, parent):
        """Create the log section"""
        log_frame = ttk.LabelFrame(parent, text="Log", padding="10")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=6, width=70)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_button_section(self, parent):
        """Create the control buttons section"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=8, column=0, columnspan=3, pady=(10, 0))
        
        self.process_button = ttk.Button(button_frame, text="Process Images", 
                                       command=self.controller.start_processing, style="Accent.TButton")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Open Output Folder", 
                  command=self.controller.open_output_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.RIGHT)
    
    def update_images_display(self, images, count):
        """Update the images display"""
        if count == 0:
            self.images_label.config(text="No images selected")
        else:
            self.images_label.config(text=f"{count} images selected")
            
        # Update listbox
        self.images_listbox.delete(0, tk.END)
        for img_path in images:
            filename = os.path.basename(img_path)
            self.images_listbox.insert(tk.END, filename)
            
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_var.set(value)
        
    def update_status(self, message):
        """Update the status label"""
        self.root.after(0, lambda: self.status_label.config(text=message))
        
    def update_process_button(self, enabled, text="Process Images"):
        """Update the process button state and text"""
        state = "normal" if enabled else "disabled"
        self.process_button.config(state=state, text=text)
        
    def log_message(self, message):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
        
        self.root.after(0, update_log)
        
    def show_warning(self, title, message):
        """Show a warning message box"""
        messagebox.showwarning(title, message)
        
    def show_info(self, title, message):
        """Show an info message box"""
        messagebox.showinfo(title, message)