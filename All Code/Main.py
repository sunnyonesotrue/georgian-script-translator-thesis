import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import os
from pathlib import Path
import threading
from PIL import Image, ImageTk, ImageDraw
import json
import cv2
import numpy as np
from datetime import datetime
from Asomtavruli_Class import AsomtavruliOCR
from Nuskhuri_Class import NuskhuriOCR
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
        self.generate_text_files = tk.BooleanVar(value=False)
        self.translate_to_modern = tk.BooleanVar(value=False)
        self.translate_onto_image = tk.BooleanVar(value=False)   # NEW
        self.generate_original_text = tk.BooleanVar(value=False) # NEW

        # Settings
        self.settings_file = "translator_settings.json"
        self.load_settings()

        # Create UI first
        self.ui = ImageTranslatorUI(root, self)
        self.ui.log_message("UI created successfully")

        # Initialize OCR processors after UI is created
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

        if self.asomtavruli_ocr is None and self.nuskhuri_ocr is None:
            self.ui.log_message("WARNING: No OCR processors initialized - will run in fallback mode")
        else:
            status_msgs = []
            if self.asomtavruli_ocr:
                status_msgs.append("Asomtavruli")
            if self.nuskhuri_ocr:
                status_msgs.append("Nuskhuri")
            self.ui.log_message(f"OCR processors successfully initialized: {', '.join(status_msgs)}")

        self.ui.log_message("Application started")

    # ------------------------------------------------------------------
    # OCR initialisation
    # ------------------------------------------------------------------

    def initialize_ocr(self):
        """Initialize both OCR processors with their respective paths."""

        self.ui.log_message("=== OCR INITIALIZATION DEBUG ===")
        self.ui.log_message("Step 1: initialize_ocr() method called")

        # Asomtavruli
        self.ui.log_message("Step 2: Initializing Asomtavruli OCR...")
        try:
            asomtavruli_model_path = (
                "/Users/sunnysideup/Documents/GitHub/Georgian-Script-Translator-Thesis/Asomtavruli Data/Neural Networks/best_dynamic_model_try10_97.60.pth"
            )
            asomtavruli_data_path = (
                "/Users/sunnysideup/Documents/GitHub/Georgian-Script-Translator-Thesis/Asomtavruli Data/Sorted"
            )
            font_path = (
                "/Users/sunnysideup/Documents/GitHub/Georgian-Script-Translator-Thesis/Asomtavruli Data/NotoSansGeorgian-VariableFont_wdth,wght.ttf"
            )
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
                    image_size=64,
                )
                self.ui.log_message("✓ Asomtavruli OCR initialized successfully")
            else:
                self.ui.log_message("✗ Asomtavruli files not found:")
                self.ui.log_message(
                    f"  Model: {asomtavruli_model_path} "
                    f"({'Found' if os.path.exists(asomtavruli_model_path) else 'Not Found'})"
                )
                self.ui.log_message(
                    f"  Data: {asomtavruli_data_path} "
                    f"({'Found' if os.path.exists(asomtavruli_data_path) else 'Not Found'})"
                )
        except Exception as e:
            self.ui.log_message(f"✗ Asomtavruli OCR initialization failed: {str(e)}")
            self.asomtavruli_ocr = None

        # Nuskhuri
        self.ui.log_message("Step 3: Initializing Nuskhuri OCR...")
        try:
            nuskhuri_model_path = (
                "/Users/sunnysideup/Documents/GitHub/Georgian-Script-Translator-Thesis/Nuskhuri Data/Neural Networks/Legacy Models_99.17.pth"
            )
            nuskhuri_data_path = (
                "/Users/sunnysideup/Documents/GitHub/Georgian-Script-Translator-Thesis/Nuskhuri Data/Sorted"
            )
            font_path = (
                "/Users/sunnysideup/Documents/GitHub/Georgian-Script-Translator-Thesis/Asomtavruli Data/NotoSansGeorgian-VariableFont_wdth,wght.ttf"
            )
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
                    image_size=64,
                )
                self.ui.log_message("✓ Nuskhuri OCR initialized successfully")
            else:
                self.ui.log_message("✗ Nuskhuri files not found:")
                self.ui.log_message(
                    f"  Model: {nuskhuri_model_path} "
                    f"({'Found' if os.path.exists(nuskhuri_model_path) else 'Not Found'})"
                )
                self.ui.log_message(
                    f"  Data: {nuskhuri_data_path} "
                    f"({'Found' if os.path.exists(nuskhuri_data_path) else 'Not Found'})"
                )
        except Exception as e:
            self.ui.log_message(f"✗ Nuskhuri OCR initialization failed: {str(e)}")
            self.nuskhuri_ocr = None

        self.ui.log_message("=== OCR INITIALIZATION COMPLETE ===")

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def select_images(self):
        """Handle image / PDF selection via file dialog."""
        filetypes = (
            ("Images & PDFs", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.pdf"),
            ("PDF files", "*.pdf"),
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*"),
        )

        files = filedialog.askopenfilenames(
            title="Select Images or PDFs",
            initialdir=os.getcwd(),
            filetypes=filetypes,
        )

        if files:
            self.selected_images = list(files)
            self.update_images_display()
            self.ui.log_message(f"Selected {len(self.selected_images)} file(s)")

    def clear_images(self):
        """Clear the selected images."""
        self.selected_images = []
        self.update_images_display()
        self.ui.log_message("Cleared selected images")

    def on_translation_source_change(self):
        """Handle translation source change."""
        source = self.translation_source.get()
        source_name = "Asomtavruli" if source == "asomtavruli" else "Nuskhuri"
        self.ui.log_message(f"Translation source changed to: {source_name}")
        self.save_settings()

    def update_images_display(self):
        """Update the images display in UI."""
        count = len(self.selected_images)
        self.ui.update_images_display(self.selected_images, count)

    def select_output_directory(self):
        """Handle output directory selection."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_directory.get() or os.getcwd(),
        )

        if directory:
            self.output_directory.set(directory)
            self.save_settings()
            self.ui.log_message(f"Output directory set to: {directory}")

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def start_processing(self):
        """Start the image/PDF processing."""
        if not self.selected_images:
            self.ui.show_warning("No Files", "Please select images or PDFs to process.")
            return

        if not self.output_directory.get():
            self.ui.show_warning("No Output Directory", "Please select an output directory.")
            return

        if self.processing:
            self.ui.show_info("Processing", "Already processing. Please wait.")
            return

        self.processing = True
        self.ui.update_process_button(False, "Processing...")

        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()

    def process_images_thread(self):
        """Process files in a separate thread."""
        try:
            total = len(self.selected_images)
            processed = 0
            failed = 0

            self.ui.log_message(f"Starting batch processing of {total} file(s)...")

            for i, file_path in enumerate(self.selected_images):
                try:
                    progress = (i / total) * 100
                    self.ui.update_progress(progress)

                    filename = os.path.basename(file_path)
                    file_type = "PDF" if file_path.lower().endswith(".pdf") else "image"
                    self.ui.update_status(
                        f"Processing {i+1}/{total}: {filename} ({file_type}, {self.translation_source.get()})..."
                    )
                    self.ui.log_message(f"Processing {file_type} {i+1}/{total}: {filename}")

                    success = self.process_single_image(file_path)

                    if success:
                        processed += 1
                        self.ui.log_message(f"✓ Successfully processed: {filename}")
                    else:
                        failed += 1
                        self.ui.log_message(f"✗ Failed to process: {filename}")

                except Exception as e:
                    failed += 1
                    self.ui.log_message(
                        f"✗ Exception processing {os.path.basename(file_path)}: {str(e)}"
                    )
                    continue

            self.ui.update_progress(100)
            self.ui.update_status(
                f"Batch complete! Processed: {processed}, Failed: {failed}"
            )
            self.ui.log_message(
                f"Batch processing complete: {processed} successful, {failed} failed out of {total} total"
            )

        except Exception as e:
            self.ui.log_message(f"Critical error during batch processing: {str(e)}")
            self.ui.update_status("Critical error occurred during processing")
            import traceback
            self.ui.log_message(f"Full traceback: {traceback.format_exc()}")

        finally:
            self.processing = False
            self.ui.update_process_button(True)

    def process_single_image(self, file_path):
        """
        Route a single file to the appropriate processor.
        PDFs get their own pipeline; everything else uses the existing image pipeline.
        """
        if file_path.lower().endswith(".pdf"):
            return self.process_pdf_file(file_path)
        return self._process_image_file(file_path)

    # ------------------------------------------------------------------
    # Image processing (unchanged from original)
    # ------------------------------------------------------------------

    def _process_image_file(self, image_path):
        """
        Process a single image using the selected OCR engine.
        Uses run_on_all_thresholds to produce multiple variants.
        """
        try:
            source_script = self.translation_source.get()

            if source_script == "asomtavruli":
                ocr = self.asomtavruli_ocr
                ocr_name = "Asomtavruli"
            else:
                ocr = self.nuskhuri_ocr
                ocr_name = "Nuskhuri"

            if ocr is None:
                fallback_ocr = (
                    self.asomtavruli_ocr if source_script != "asomtavruli" else self.nuskhuri_ocr
                )
                fallback_name = (
                    "Asomtavruli" if source_script != "asomtavruli" else "Nuskhuri"
                )
                if fallback_ocr is not None:
                    self.ui.log_message(
                        f"{ocr_name} OCR not available. Falling back to {fallback_name}."
                    )
                    ocr = fallback_ocr
                    ocr_name = fallback_name
                else:
                    self.ui.log_message(
                        "No OCR engines initialized. Running in placeholder fallback mode..."
                    )
                    return True

            ocr.output_dir = self.output_directory.get()
            os.makedirs(ocr.output_dir, exist_ok=True)

            filename = os.path.basename(image_path)
            self.ui.log_message(
                f"Running {ocr_name} OCR with all thresholds on {filename}..."
            )

            saved_paths = ocr.run_on_all_thresholds(
                image_path,
                show=False,
                generate_text=self.generate_text_files.get(),
                translate_text=self.translate_to_modern.get(),
                translate_onto_image=self.translate_onto_image.get(),
                generate_original_text=self.generate_original_text.get(),
            )

            self.ui.log_message(f"Generated {len(saved_paths)} threshold variants:")
            for idx, out_path in enumerate(saved_paths, start=1):
                self.ui.log_message(f"  {idx}. {os.path.basename(out_path)}")

            return True

        except Exception as e:
            self.ui.log_message(
                f"Error processing {os.path.basename(image_path)}: {str(e)}"
            )
            import traceback
            self.ui.log_message(f"Full error: {traceback.format_exc()}")
            return False

    # ------------------------------------------------------------------
    # PDF processing  (NEW)
    # ------------------------------------------------------------------

    def _resolve_ocr(self):
        """
        Return (ocr_instance, ocr_name) based on user selection,
        with automatic fallback if the chosen engine is not loaded.
        Returns (None, None) if no engine is available.
        """
        source_script = self.translation_source.get()

        if source_script == "asomtavruli":
            ocr, ocr_name = self.asomtavruli_ocr, "Asomtavruli"
            fallback, fallback_name = self.nuskhuri_ocr, "Nuskhuri"
        else:
            ocr, ocr_name = self.nuskhuri_ocr, "Nuskhuri"
            fallback, fallback_name = self.asomtavruli_ocr, "Asomtavruli"

        if ocr is None and fallback is not None:
            self.ui.log_message(
                f"{ocr_name} OCR not available. Falling back to {fallback_name}."
            )
            return fallback, fallback_name

        return ocr, ocr_name

    def _annotate_gray_array(self, gray_array, boxes, preds, ocr,
                              translate_onto_image: bool = False):
        """
        Draw OCR results on a grayscale numpy array and return a PIL RGB image.
        When translate_onto_image is True, labels are swapped to their modern
        Georgian equivalents using the OCR engine's own translation map.
        """
        # Grab whichever translation map the OCR object exposes
        translation_map = (
            getattr(ocr, "asomtavruli_to_modern", None)
            or getattr(ocr, "nuskhuri_to_modern", {})
        )

        rgb = cv2.cvtColor(gray_array, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        font = ocr._get_font(size=20)
        skip_labels = {","}

        for (x, y, w, h), pred in zip(boxes, preds):
            if pred in skip_labels:
                continue
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
            text_y = y - 15 if y - 15 >= 0 else y + h + 2
            label = (
                translation_map.get(pred, pred)
                if translate_onto_image
                else pred
            )
            draw.text((x, text_y), label, fill="green", font=font)

        return pil_img

    def _get_threshold_variants(self, gray, ocr_name):
        """
        Return an ordered dict  {variant_name: binary_numpy_array}  for every
        thresholding technique that matches the active OCR engine.

        Asomtavruli → ThresholdManager.threshold_variants_from_image()  (list)
        Nuskhuri    → ThresholdManager.run_all_Nuskuri_Thresholds()      (dict)
        """
        from ThresholdManager import ThresholdManager as TM
        tm = TM()

        if ocr_name == "Asomtavruli":
            variant_list = tm.threshold_variants_from_image(gray)
            names = [
                "mean_median", "mean_closing", "gaussian_median",
                "gaussian_closing", "otsu", "mean_bilateral",
                "gaussian_bilateral", "mean_NLMD", "gaussian_NLMD", "all",
            ]
            # Pad names if ThresholdManager ever returns extra variants
            if len(names) < len(variant_list):
                names += [f"var_{i}" for i in range(len(names), len(variant_list))]
            return dict(zip(names[: len(variant_list)], variant_list))

        else:  # Nuskhuri
            return tm.run_all_Nuskuri_Thresholds(gray)

    def process_pdf_file(self, pdf_path):
        """
        Convert each PDF page to a grayscale image, apply every threshold
        variant, run OCR on each variant, annotate the original page, and
        save one output PDF per threshold variant.

        Naming convention (mirrors the image pipeline):
            <ocr_name>_ocr_<base_filename>_<variant_name>.pdf

        When "Generate text files" is enabled, one .txt file is also written
        per variant:
            <ocr_name>_text_<base_filename>_<variant_name>.txt

        Requires PyMuPDF  →  pip3 install pymupdf
        """
        try:
            try:
                import fitz  # PyMuPDF
            except ImportError:
                self.ui.log_message(
                    "✗ PyMuPDF is not installed. "
                    "Install it with:  pip3 install pymupdf"
                )
                return False

            ocr, ocr_name = self._resolve_ocr()
            if ocr is None:
                self.ui.log_message("✗ No OCR engine available for PDF processing.")
                return False

            ocr.output_dir = self.output_directory.get()
            os.makedirs(ocr.output_dir, exist_ok=True)

            base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            self.ui.log_message(
                f"PDF '{os.path.basename(pdf_path)}' — {num_pages} page(s), "
                f"{ocr_name} OCR, all threshold variants..."
            )

            # Accumulators keyed by variant name
            # variant_pages[name] = [PIL_page0, PIL_page1, ...]
            # variant_texts[name] = ["=== Page 1 ===\n...", ...]
            variant_pages = {}   # {str: list[PIL.Image]}
            variant_texts = {}   # {str: list[str]}

            # ── Process every page ────────────────────────────────────────
            for page_idx in range(num_pages):
                page = doc[page_idx]
                page_label = f"page {page_idx + 1}/{num_pages}"

                # Render at 200 DPI → grayscale numpy array
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.h, pix.w
                )
                self.ui.log_message(
                    f"  {page_label} ({pix.w}×{pix.h} px) — "
                    f"computing threshold variants..."
                )

                # Get all threshold variants for this page
                try:
                    variants = self._get_threshold_variants(gray, ocr_name)
                except Exception as e:
                    self.ui.log_message(
                        f"  ✗ {page_label}: could not compute variants: {e}"
                    )
                    continue

                # ── For each variant, run OCR and accumulate results ──────
                for variant_name, binary in variants.items():
                    if not isinstance(binary, np.ndarray):
                        self.ui.log_message(
                            f"  ✗ {page_label}/{variant_name}: not a valid array, skipping"
                        )
                        continue

                    try:
                        boxes, preds = ocr.run_on_array(binary)
                        char_count = sum(1 for p in preds if p != ",")
                        self.ui.log_message(
                            f"  {page_label} [{variant_name}]: "
                            f"{char_count} character(s) detected"
                        )

                        # Annotate the *original* (un-thresholded) page
                        annotated_pil = self._annotate_gray_array(
                            gray, boxes, preds, ocr,
                            translate_onto_image=self.translate_onto_image.get(),
                        )

                        if variant_name not in variant_pages:
                            variant_pages[variant_name] = []
                        variant_pages[variant_name].append(annotated_pil)

                        # ── Translated text file (existing option) ────────
                        if self.generate_text_files.get() and preds:
                            page_text = ocr.generate_text_from_boxes(
                                boxes, preds, gray.shape,
                                self.translate_to_modern.get(),
                            )
                            if page_text:
                                entry = f"=== Page {page_idx + 1} ===\n{page_text}"
                                if variant_name not in variant_texts:
                                    variant_texts[variant_name] = []
                                variant_texts[variant_name].append(entry)

                        # ── Original-script text file (new option) ────────
                        if self.generate_original_text.get() and preds:
                            orig_text = ocr.generate_text_from_boxes(
                                boxes, preds, gray.shape,
                                translate_text=False,
                            )
                            if orig_text:
                                entry = f"=== Page {page_idx + 1} ===\n{orig_text}"
                                key = variant_name + "__original"
                                if key not in variant_texts:
                                    variant_texts[key] = []
                                variant_texts[key].append(entry)

                    except Exception as e:
                        self.ui.log_message(
                            f"  ✗ {page_label}/{variant_name}: OCR error — {e}"
                        )
                        continue

            doc.close()

            # ── Save one PDF (and optionally text files) per variant ─────
            saved_pdfs = 0
            for variant_name, pages in variant_pages.items():
                if not pages:
                    continue

                pdf_name = (
                    f"{ocr_name.lower()}_ocr_{base_filename}_{variant_name}.pdf"
                )
                pdf_out = os.path.join(ocr.output_dir, pdf_name)

                pages[0].save(
                    pdf_out,
                    save_all=True,
                    append_images=pages[1:],
                )
                self.ui.log_message(
                    f"  ✓ Saved [{variant_name}] — {len(pages)} page(s): {pdf_name}"
                )
                saved_pdfs += 1

                # Translated text file (existing option)
                if self.generate_text_files.get() and variant_name in variant_texts:
                    txt_name = (
                        f"{ocr_name.lower()}_text_{base_filename}_{variant_name}_translated.txt"
                    )
                    txt_out = os.path.join(ocr.output_dir, txt_name)
                    with open(txt_out, "w", encoding="utf-8") as fh:
                        fh.write("\n\n".join(variant_texts[variant_name]))
                    self.ui.log_message(
                        f"  ✓ Saved translated text [{variant_name}]: {txt_name}"
                    )

                # Original-script text file (new option)
                orig_key = variant_name + "__original"
                if self.generate_original_text.get() and orig_key in variant_texts:
                    txt_name = (
                        f"{ocr_name.lower()}_text_{base_filename}_{variant_name}_original.txt"
                    )
                    txt_out = os.path.join(ocr.output_dir, txt_name)
                    with open(txt_out, "w", encoding="utf-8") as fh:
                        fh.write("\n\n".join(variant_texts[orig_key]))
                    self.ui.log_message(
                        f"  ✓ Saved original text [{variant_name}]: {txt_name}"
                    )

            self.ui.log_message(
                f"✓ PDF processing complete: {saved_pdfs} variant PDF(s) saved "
                f"for '{os.path.basename(pdf_path)}'"
            )
            return True

        except Exception as e:
            self.ui.log_message(
                f"✗ Error processing PDF '{os.path.basename(pdf_path)}': {str(e)}"
            )
            import traceback
            self.ui.log_message(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def open_output_folder(self):
        """Open the output folder in the file explorer."""
        if self.output_directory.get() and os.path.exists(self.output_directory.get()):
            if os.name == "nt":
                os.startfile(self.output_directory.get())
            elif os.name == "posix":
                import sys
                os.system(
                    f'open "{self.output_directory.get()}"'
                    if sys.platform == "darwin"
                    else f'xdg-open "{self.output_directory.get()}"'
                )
        else:
            self.ui.show_warning(
                "Invalid Directory", "Output directory not set or doesn't exist."
            )

    def save_settings(self):
        """Save application settings."""
        settings = {
            "output_directory": self.output_directory.get(),
            "translation_source": self.translation_source.get(),
            "generate_text_files": self.generate_text_files.get(),
            "translate_to_modern": self.translate_to_modern.get(),
            "translate_onto_image": self.translate_onto_image.get(),
            "generate_original_text": self.generate_original_text.get(),
        }
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f)
        except Exception as e:
            self.ui.log_message(f"Failed to save settings: {str(e)}")

    def load_settings(self):
        """Load application settings."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    self.output_directory.set(settings.get("output_directory", ""))
                    self.translation_source.set(
                        settings.get("translation_source", "asomtavruli")
                    )
                    self.generate_text_files.set(
                        settings.get("generate_text_files", False)
                    )
                    self.translate_to_modern.set(
                        settings.get("translate_to_modern", False)
                    )
                    self.translate_onto_image.set(
                        settings.get("translate_onto_image", False)
                    )
                    self.generate_original_text.set(
                        settings.get("generate_original_text", False)
                    )
        except Exception:
            pass  # Use defaults if loading fails


def main():
    """Main application entry point."""
    root = tk.Tk()
    app = ImageTranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    main()