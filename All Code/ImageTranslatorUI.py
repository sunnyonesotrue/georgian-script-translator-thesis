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
        self.root.geometry("800x780")
        self.root.minsize(650, 600)

        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")

    def setup_ui(self):
        """Setup the user interface using a vertical PanedWindow so the log
        panel can be resized by dragging the sash."""

        # ── Outer shell ───────────────────────────────────────────────────
        outer = ttk.Frame(self.root, padding="10")
        outer.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)   # paned window expands
        outer.rowconfigure(1, weight=0)   # button bar stays fixed

        # ── Vertical PanedWindow ──────────────────────────────────────────
        paned = ttk.PanedWindow(outer, orient=tk.VERTICAL)
        paned.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ── TOP PANE: all fixed controls ──────────────────────────────────
        top_pane = ttk.Frame(paned, padding=(0, 0, 0, 4))
        paned.add(top_pane, weight=0)   # weight=0 → top gets minimal growth
        top_pane.columnconfigure(1, weight=1)

        self.create_title_section(top_pane)
        self.create_input_section(top_pane)
        self.create_translation_source_section(top_pane)
        self.create_output_section(top_pane)
        self.create_text_generation_section(top_pane)
        self.create_preview_section(top_pane)
        self.create_progress_section(top_pane)

        # ── BOTTOM PANE: log ──────────────────────────────────────────────
        log_pane = ttk.Frame(paned, padding=(0, 4, 0, 0))
        paned.add(log_pane, weight=1)   # weight=1 → log absorbs extra space
        log_pane.columnconfigure(0, weight=1)
        log_pane.rowconfigure(0, weight=1)

        self.create_log_section(log_pane)

        # ── BUTTON BAR: always visible below the sash ─────────────────────
        self.create_button_section(outer)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def create_title_section(self, parent):
        """Create the title section"""
        title_label = ttk.Label(
            parent, text="Image OCR Translator", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

    def create_input_section(self, parent):
        """Create the input images / PDFs section"""
        input_frame = ttk.LabelFrame(
            parent, text="Input Files (Images & PDFs)", padding="10"
        )
        input_frame.grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        input_frame.columnconfigure(1, weight=1)

        ttk.Button(
            input_frame,
            text="Select Files",
            command=self.controller.select_images,
        ).grid(row=0, column=0, padx=(0, 10))

        self.images_label = ttk.Label(input_frame, text="No files selected")
        self.images_label.grid(row=0, column=1, sticky=tk.W)

        ttk.Button(
            input_frame, text="Clear", command=self.controller.clear_images
        ).grid(row=0, column=2)

        # Small hint about accepted formats
        hint = ttk.Label(
            input_frame,
            text="Accepted: PNG, JPG, BMP, TIFF, GIF, PDF  "
                 "—  PDFs are processed page-by-page and saved as an annotated PDF",
            foreground="grey",
            font=("Arial", 8),
        )
        hint.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

    def create_translation_source_section(self, parent):
        """Create the translation source selection section"""
        translation_frame = ttk.LabelFrame(
            parent, text="Translation Source", padding="10"
        )
        translation_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        ttk.Label(translation_frame, text="Translate from:").grid(
            row=0, column=0, padx=(0, 15), sticky=tk.W
        )

        ttk.Radiobutton(
            translation_frame,
            text="Asomtavruli (ⴀⴑⴍⴋⴇⴀⴅⴐⴓⴊⴈ)",
            variable=self.controller.translation_source,
            value="asomtavruli",
            command=self.controller.on_translation_source_change,
        ).grid(row=0, column=1, padx=(0, 20), sticky=tk.W)

        ttk.Radiobutton(
            translation_frame,
            text="Nuskhuri (ⴌⴓⴑⴞⴓⴐⴈ)",
            variable=self.controller.translation_source,
            value="nuskhuri",
            command=self.controller.on_translation_source_change,
        ).grid(row=0, column=2, sticky=tk.W)

    def create_output_section(self, parent):
        """Create the output directory section"""
        output_frame = ttk.LabelFrame(parent, text="Output Location", padding="10")
        output_frame.grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        output_frame.columnconfigure(1, weight=1)

        ttk.Button(
            output_frame,
            text="Select Output Folder",
            command=self.controller.select_output_directory,
        ).grid(row=0, column=0, padx=(0, 10))

        ttk.Entry(
            output_frame,
            textvariable=self.controller.output_directory,
            state="readonly",
        ).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

    def create_text_generation_section(self, parent):
        """Create the output options section"""
        text_gen_frame = ttk.LabelFrame(parent, text="Output Options", padding="10")
        text_gen_frame.grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        # ── Image output ──────────────────────────────────────────────────
        img_label = ttk.Label(text_gen_frame, text="Image output:",
                              font=("Arial", 9, "bold"))
        img_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 2))

        self.translate_onto_image_checkbox = ttk.Checkbutton(
            text_gen_frame,
            text="Overlay Modern Georgian translation on annotated images "
                 "(replaces original script labels in the output image)",
            variable=self.controller.translate_onto_image,
            command=self.controller.save_settings,
        )
        self.translate_onto_image_checkbox.grid(row=1, column=0, sticky=tk.W, padx=(12, 0))

        # ── Text output ───────────────────────────────────────────────────
        txt_label = ttk.Label(text_gen_frame, text="Text output:",
                              font=("Arial", 9, "bold"))
        txt_label.grid(row=2, column=0, sticky=tk.W, pady=(8, 2))

        self.generate_original_text_checkbox = ttk.Checkbutton(
            text_gen_frame,
            text="Generate text file — original script characters  "
                 "(*_original.txt)",
            variable=self.controller.generate_original_text,
            command=self.controller.save_settings,
        )
        self.generate_original_text_checkbox.grid(row=3, column=0, sticky=tk.W, padx=(12, 0))

        self.text_gen_checkbox = ttk.Checkbutton(
            text_gen_frame,
            text="Generate text file — spatially preserved OCR results  "
                 "(*_translated.txt when translated, otherwise plain)",
            variable=self.controller.generate_text_files,
            command=self.controller.save_settings,
        )
        self.text_gen_checkbox.grid(row=4, column=0, sticky=tk.W, padx=(12, 0))

        self.text_gen_translate_checkbox = ttk.Checkbutton(
            text_gen_frame,
            text="Translate text file output to Modern Georgian",
            variable=self.controller.translate_to_modern,
            command=self.controller.save_settings,
        )
        self.text_gen_translate_checkbox.grid(row=5, column=0, sticky=tk.W, padx=(24, 0))

    def create_preview_section(self, parent):
        """Create the selected files preview section"""
        preview_frame = ttk.LabelFrame(
            parent, text="Selected Files", padding="10"
        )
        preview_frame.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky=(tk.W, tk.E, tk.N, tk.S),
            pady=(0, 10),
        )
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.images_listbox = tk.Listbox(preview_frame, height=4)
        self.images_listbox.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10)
        )

        scrollbar = ttk.Scrollbar(
            preview_frame, orient="vertical", command=self.images_listbox.yview
        )
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.images_listbox.configure(yscrollcommand=scrollbar.set)

    def create_progress_section(self, parent):
        """Create the progress section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(
            row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.grid(
            row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5)
        )

        self.status_label = ttk.Label(progress_frame, text="Ready to process files")
        self.status_label.grid(row=1, column=0, sticky=tk.W)

    def create_log_section(self, parent):
        """Create the log section — fills whatever space the sash allows."""
        log_frame = ttk.LabelFrame(parent, text="Log  (drag sash above to resize)", padding="10")
        log_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, height=6, width=70)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def create_button_section(self, parent):
        """Create the control buttons section — always visible below the sash."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, pady=(10, 0), sticky=(tk.W, tk.E))

        self.process_button = ttk.Button(
            button_frame,
            text="Process Files",
            command=self.controller.start_processing,
            style="Accent.TButton",
        )
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame,
            text="Open Output Folder",
            command=self.controller.open_output_folder,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame, text="Exit", command=self.root.quit
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Update helpers (called from controller)
    # ------------------------------------------------------------------

    def update_images_display(self, images, count):
        """Update the files display"""
        if count == 0:
            self.images_label.config(text="No files selected")
        else:
            pdf_count = sum(1 for p in images if p.lower().endswith(".pdf"))
            img_count = count - pdf_count
            parts = []
            if img_count:
                parts.append(f"{img_count} image(s)")
            if pdf_count:
                parts.append(f"{pdf_count} PDF(s)")
            self.images_label.config(text=", ".join(parts) + " selected")

        self.images_listbox.delete(0, tk.END)
        for file_path in images:
            filename = os.path.basename(file_path)
            prefix = "[PDF] " if file_path.lower().endswith(".pdf") else "[IMG] "
            self.images_listbox.insert(tk.END, prefix + filename)

    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_var.set(value)

    def update_status(self, message):
        """Update the status label"""
        self.root.after(0, lambda: self.status_label.config(text=message))

    def update_process_button(self, enabled, text="Process Files"):
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