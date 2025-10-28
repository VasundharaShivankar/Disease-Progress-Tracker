import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import sys
import io
import os # <-- os import is crucial and correctly placed

# --- Import Core Analysis Logic ---
try:
    # Ensure segment_lesion now accepts the 'disease' argument
    from src.skin_analysis import segment_lesion, calculate_lesion_area
except ImportError:
    messagebox.showerror("Error", "Could not import core analysis logic from src/skin_analysis.py.\nPlease ensure the file exists and is in the 'src' directory.")
    sys.exit()


# --- Define Hospital/Clinic Color Palette ---
COLOR_PRIMARY = '#007ACC'  # Professional Blue
COLOR_SECONDARY = '#004C8C' # Darker Blue for accents
COLOR_BACKGROUND = '#F0F5F9' # Light Gray/Off-White for background
COLOR_SUCCESS = '#28A745'   # Green for success/progress
COLOR_ERROR = '#DC3545'     # Red for errors/regression
COLOR_TEXT = '#343A40'      # Dark Gray for main text


class SkinTrackerApp:
    def __init__(self, master):
        self.master = master
        master.title("DermAI Progress Tracker")
        master.config(bg=COLOR_BACKGROUND)

        self.past_image_path = None
        self.new_image_path = None

        # --- Define Diseases and Selection Variable ---
        self.diseases = ["Skin Lesion (Generic/Acne)", "Nail Psoriasis", "Dermatitis / Eczema", "Stevens-Johnson Syndrome (SJS)"]
        self.selected_disease = tk.StringVar(master)
        self.selected_disease.set(self.diseases[0]) # Default value is Generic Lesion

        # --- Apply Modern Theme ---
        style = ttk.Style()
        style.theme_use('clam') # 'clam' is a modern theme option in Tkinter
        style.configure('TFrame', background=COLOR_BACKGROUND)
        style.configure('TLabel', background=COLOR_BACKGROUND, foreground=COLOR_TEXT, font=('Segoe UI', 10))
        style.configure('TButton', background=COLOR_PRIMARY, foreground='white', font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('TButton', background=[('active', COLOR_SECONDARY)])
        
        # Configure Grid Layout
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(5, weight=1) # Row 5 holds the result text

        # --- Header ---
        header_frame = ttk.Frame(master, padding="15 10 15 10", style='TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew')
        tk.Label(header_frame, text="DermAI: Clinical Lesion Progress Tracking", bg=COLOR_PRIMARY, fg='white', font=('Segoe UI', 16, 'bold'), anchor='center').pack(fill='x')

        # --- Main Content Frame ---
        main_frame = ttk.Frame(master, padding="20", style='TFrame')
        main_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # --- NEW: Disease Selector ---
        tk.Label(main_frame, text="Select Disease Type for Analysis:", bg=COLOR_BACKGROUND, fg=COLOR_TEXT, font=('Segoe UI', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky='w', padx=10)
        
        # Dropdown Menu
        option_menu = ttk.OptionMenu(main_frame, self.selected_disease, self.diseases[0], *self.diseases)
        option_menu.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky='ew', padx=100)
        style.configure("TMenubutton", font=('Segoe UI', 11))

        # 1. Past Image Controls
        tk.Label(main_frame, text="1. Baseline Image (Past)", bg=COLOR_BACKGROUND, fg=COLOR_TEXT, font=('Segoe UI', 12, 'bold')).grid(row=1, column=0, pady=10)
        self.past_img_label = tk.Label(main_frame, text="Click to Select", width=40, height=15, bg='white', fg='gray', borderwidth=1, relief="solid")
        self.past_img_label.grid(row=2, column=0, padx=10, pady=5, sticky='nsew')
        ttk.Button(main_frame, text="Select Past Image 📂", command=lambda: self.select_image('past')).grid(row=3, column=0, pady=10)
        
        # 2. New Image Controls
        tk.Label(main_frame, text="2. Follow-up Image (New)", bg=COLOR_BACKGROUND, fg=COLOR_TEXT, font=('Segoe UI', 12, 'bold')).grid(row=1, column=1, pady=10)
        self.new_img_label = tk.Label(main_frame, text="Click to Select", width=40, height=15, bg='white', fg='gray', borderwidth=1, relief="solid")
        self.new_img_label.grid(row=2, column=1, padx=10, pady=5, sticky='nsew')
        ttk.Button(main_frame, text="Select New Image 📂", command=lambda: self.select_image('new')).grid(row=3, column=1, pady=10)

        # 3. Analysis Button
        ttk.Button(master, text="✨ ANALYZE PROGRESS ✨", command=self.run_analysis, style='Analyze.TButton').grid(row=2, column=0, columnspan=2, pady=(10, 20), ipadx=30, ipady=10)
        style.configure('Analyze.TButton', background=COLOR_SUCCESS, foreground='white', font=('Segoe UI', 14, 'bold'))
        style.map('Analyze.TButton', background=[('active', COLOR_SECONDARY)])

        # 4. Results Area
        tk.Label(master, text="--- Quantitative Results ---", bg=COLOR_BACKGROUND, fg=COLOR_PRIMARY, font=('Segoe UI', 12, 'bold')).grid(row=3, column=0, columnspan=2, pady=(10, 5))
        self.result_text = tk.Text(master, height=12, width=85, state='disabled', wrap='word', font=('Consolas', 10), bg='white', fg=COLOR_TEXT, borderwidth=1, relief="sunken")
        self.result_text.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 20), sticky='nsew')


    def select_image(self, type):
        """Opens a file dialog and updates the image label with a thumbnail."""
        file_path = filedialog.askopenfilename(
            title=f"Select {type.capitalize()} Lesion Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                img = Image.open(file_path)
                # Resize for display thumbnail
                img.thumbnail((300, 200)) 
                tk_img = ImageTk.PhotoImage(img)
            except Exception as e:
                messagebox.showerror("Image Error", f"Could not load image: {e}")
                return

            if type == 'past':
                self.past_image_path = file_path
                self.past_img_label.config(image=tk_img, text="", bg='white')
                self.past_img_label.image = tk_img
            elif type == 'new':
                self.new_image_path = file_path
                self.new_img_label.config(image=tk_img, text="", bg='white')
                self.new_img_label.image = tk_img

    def update_result_text(self, text):
        """Updates the read-only result text box."""
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state='disabled')
        self.master.update()

    def run_analysis(self):
        """Loads images, runs the segmentation logic, and displays results."""
        if not self.past_image_path or not self.new_image_path:
            messagebox.showwarning("Warning", "Please select both Past and New images before analyzing.")
            return

        try:
            # Get the selected disease to pass to the core logic
            disease_type = self.selected_disease.get()
            
            img_past = cv2.imread(self.past_image_path)
            img_new = cv2.imread(self.new_image_path)

            if img_past is None or img_new is None:
                raise FileNotFoundError("Could not read image data. Check file corruption.")

            self.update_result_text(f"Analysis in progress for: {disease_type}...\nRunning segmentation and calculation...")

            # --- CORE ANALYSIS LOGIC ---
            # PASS THE DISEASE TYPE!
            mask_past, count_past = segment_lesion(img_past, disease=disease_type) 
            mask_new, count_new = segment_lesion(img_new, disease=disease_type) 

            # 3. Measurement
            area_past = calculate_lesion_area(mask_past)
            area_new = calculate_lesion_area(mask_new)

            # 4. Progress Calculation and Reporting
            report = io.StringIO()
            report.write("-----------------------------------------------------------\n")
            report.write(f"| DISEASE: {disease_type} |\n")
            report.write(f"| PAST: {os.path.basename(self.past_image_path)} | NEW: {os.path.basename(self.new_image_path)} |\n") 
            report.write("-----------------------------------------------------------\n")

            if count_past > 0 or count_new > 0:
                report.write(f"Lesion Count (Past/New): {count_past} / {count_new}\n")
            
            report.write(f"Lesion Area (Past/New): {area_past} / {area_new} pixels\n")
            
            # --- PROGRESS SUMMARY ---
            report.write("\n================ PROGRESS SUMMARY ================\n")

            # Area Progress
            if area_past > 0:
                percent_area_change = ((area_new - area_past) / area_past) * 100
                status_area, color_area = self._get_progress_status(percent_area_change)
                report.write(f"Area Change: {status_area} by: {abs(percent_area_change):.2f}%\n")
            
            # Count Progress
            if count_past > 0:
                percent_count_change = ((count_new - count_past) / count_past) * 100
                status_count, color_count = self._get_progress_status(percent_count_change)
                report.write(f"Count Change: {status_count} by: {abs(percent_count_change):.2f}%\n")
            
            elif area_past == 0:
                report.write("Status: Could not find a reliable lesion baseline.\n")

            report.write("==================================================\n")

            # Display results in the GUI text box
            self.update_result_text(report.getvalue())
            
            # Display Masked Visualization
            self.show_visualization(img_past, img_new, mask_past, mask_new)


        except Exception as e:
            error_message = f"An unexpected error occurred during analysis: {e}"
            messagebox.showerror("Analysis Error", error_message)
            self.update_result_text(f"ERROR: {error_message}")

    def _get_progress_status(self, percent_change):
        """Helper to determine status string and color."""
        if percent_change < -0.5:
            return "IMPROVEMENT (Decreased)", COLOR_SUCCESS
        elif percent_change > 0.5:
            return "REGRESSION (Increased)", COLOR_ERROR
        else:
            return "NO CHANGE", COLOR_TEXT

    def show_visualization(self, img_past, img_new, mask_past, mask_new):
        """Displays the masked image comparison using Matplotlib."""

        # Overlay the masks on the images for visual verification
        img_past_masked = cv2.bitwise_and(img_past, img_past, mask=mask_past)
        img_new_masked = cv2.bitwise_and(img_new, img_new, mask=mask_new)

        # Resize all images to the same height (300 pixels) to ensure compatibility for hstack
        target_height = 300
        def resize_to_height(img, height):
            h, w = img.shape[:2]
            aspect_ratio = w / h
            new_w = int(height * aspect_ratio)
            return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_LINEAR)

        img_past_resized = resize_to_height(img_past, target_height)
        img_new_resized = resize_to_height(img_new, target_height)
        img_past_masked_resized = resize_to_height(img_past_masked, target_height)
        img_new_masked_resized = resize_to_height(img_new_masked, target_height)

        # Combine and convert to RGB for Matplotlib display
        combined_image = np.hstack((img_past_resized, img_new_resized, img_past_masked_resized, img_new_masked_resized))
        combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(25, 12))
        plt.imshow(combined_image_rgb)
        plt.title("DermAI Visual Progress: Past Image | New Image | Past Masked | New Masked", fontsize=16)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    app = SkinTrackerApp(root)
    root.mainloop()