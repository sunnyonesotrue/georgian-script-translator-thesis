from ctypes import Array
from enum import unique
from re import X
from typing import Self
import cv2
import numpy as np
import os
import time
import uuid
from math import ceil


class ContourDetector:
    def __init__(self, min_area=100, output_dir="contours"):
        """
        Initializes the contour detector with a minimum area filter and output directory.
        :param min_area: Minimum area required to keep a contour.
        :param output_dir: Directory to save extracted contour images.
        """
        self.min_area = min_area
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create output directory if it doesn't exist

    def preprocess_image(self, image_path):
        """
        Reads an image, converts it to grayscale, and applies thresholding.
        :param image_path: Path to the input image.
        :return: Original image, binary image.
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        return image, binary

    def find_contours(self, binary_image):
        """
        Finds contours in a binary image.
        :param binary_image: Binary thresholded image.
        :return: List of contours.
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]  # Filter by area

    def save_contours_as_images(self, binary_image, contours, image_width, image_height, image_name):
        """
        Saves each detected contour as a separate binary image and records its attributes.
        :param binary_image: Binary thresholded image.
        :param contours: List of contours.
        :param image_width: Width of the original image.
        :param image_height: Height of the original image.
        :param image_name: Name of the original image file (used for unique naming).
        """
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = binary_image[y:y+h, x:x+w]  # Extract only the binary letter

            # Compute area and centroid
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

            # Normalize values
            norm_x = x / image_width
            norm_y = y / image_height
            norm_w = w / image_width
            norm_h = h / image_height
            norm_area = area / (image_width * image_height)
            norm_cX = cX / image_width
            norm_cY = cY / image_height

            # Generate a unique filename using UUID
            unique_id = uuid.uuid4().hex[:8]
            image_output_path = os.path.join(self.output_dir, f"{image_name}_contour_{unique_id}.png")
            txt_output_path = os.path.join(self.output_dir, f"{image_name}_contour_{unique_id}.txt")

            # Save cropped binary image
            cv2.imwrite(image_output_path, cropped)

            # Save component stats in a text file
            with open(txt_output_path, 'w') as f:
                f.write(f"Location:{image_name}\n")
                f.write(f"Component: {image_name}_contour_{unique_id}.png\n")
                f.write(f"centerX: {norm_x:.4f}\n")
                f.write(f"centerY: {norm_y:.4f}\n")
                f.write(f"width: {norm_w:.4f}\n")
                f.write(f"height: {norm_h:.4f}\n")
                f.write(f"Area: {norm_area:.4f}\n")
                f.write(f"CentroidX: {norm_cX:.4f}\n")
                f.write(f"CentroidY: {norm_cY:.4f}\n")

            print(f"Saved: {image_output_path} & {txt_output_path}")

    def process_image(self, image_path, save_contours=False):
        """
        Process a single image: Read, preprocess, detect contours, and optionally save contours.
        :param image_path: Path to the input image.
        :param save_contours: Whether to save each detected contour as a separate image.
        """
        image_name = os.path.basename(image_path).split('.')[0]  # Extract image filename without extension
        _, binary = self.preprocess_image(image_path)
        image_height, image_width = binary.shape[:2]
        contours = self.find_contours(binary)

        if save_contours:
            self.save_contours_as_images(binary, contours, image_width, image_height, image_name)

    def process_folder(self, folder_path, save_contours=True):
        """
        Processes all images in a folder that start with "hard".
        :param folder_path: Path to the folder containing images.
        :param save_contours: Whether to save detected contours as images and metadata.
        """
        if not os.path.exists(folder_path):
            print("Error: Folder does not exist.")
            return

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("No images found that start with 'hard'.")
            return
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            print(f"Processing: {image_file}")
            self.process_image(image_path, save_contours)

        print(f"Finished processing {len(image_files)} images.")
        

class SegmentationManager():
    
    inputPath = "/Users/sunnysideup/Documents/data for ORCs/All Code/Thresholding (week 2)/Thresholded images/easy1.tif_Gaussian-Closing.png"
    outputPath = "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Segmented Images/Asomtavruli"
    minimumWidthForAsomtavruli = 13
    minimumHeightForAsomtavruli = 14 #NOTE: change these to suit nuskhuri for the thesis
    
    def __init__(self, path):
        if path != "":
            self.inputPath = path
            
#NOTE: Horizontal and vertical projection profiling code
    def parse_array(self, arr):
        # Initialize result list to hold subarrays
        subarrays = []
        if not arr:
            return subarrays

        # Temporary list to hold current subarray
        current_subarray = [arr[0]]

        # Loop through the array and check for jumps
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1] + 1:  # This condition detects the jump
                subarrays.append(current_subarray)  # Add the current subarray to the result
                current_subarray = [arr[i]]  # Start a new subarray
            else:
                current_subarray.append(arr[i])  # Continue adding elements to the current subarray

        # Append the last subarray after the loop
        subarrays.append(current_subarray)
        return subarrays

    def modified_horizontal_projection(self, binary_img):
        # Get image dimensions
        height, width = binary_img.shape
    
        # List to store valid rows
        valid_rows = []
    
        for y in range(height):  # Iterate over rows, not columns
            row = binary_img[y, :]
            letter_pixels = np.sum(row == 255)  # Since we inverted, black text is 255
            non_letter_pixels = np.sum(row == 0)  # White pixels are 0

            # Adjust threshold dynamically based on document characteristics
            if letter_pixels > 10:  # Changed from 800 to a more adaptive value
                valid_rows.append(y)

        # Crop the image to only include detected rows
        returnArray = []
        if valid_rows:
            print(f"Valid rows: {valid_rows}\n")
            parsedRows = self.parse_array(valid_rows)
            print(f"Parsed valid rows: {parsedRows}\n")

            for subarray in parsedRows:
                cropped_img = binary_img[min(subarray):max(subarray) + 1, :]
                returnArray.append(cropped_img)

            print("Cropped images created")

        return returnArray

    def vertical_projection_profile(self, image):
        # Sum the pixels vertically (along columns)
        return np.sum(image == 255, axis=0)  # Ensure it counts text pixels

    def extract_letters(self, array):
        letter_images = []
        
        for image in array:
            # Get the vertical projection profile
            proj_profile = self.vertical_projection_profile(image)

            # Find the start and end positions of each letter
            letters = []
            in_letter = False
            start_col = 0

            for i in range(len(proj_profile)):
                if proj_profile[i] > 0 and not in_letter:
                    start_col = i
                    in_letter = True
                elif proj_profile[i] == 0 and in_letter:
                    end_col = i
                    letters.append((start_col, end_col))
                    in_letter = False

            # Extract individual letters
            for start_col, end_col in letters:
                letter_image = image[:, start_col:end_col]
                if letter_image.shape[1] > 1:  # Ensure it's not an empty slice
                    letter_images.append(letter_image)
        
        return letter_images

    def segment_PP(self, path, outputPath):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale before thresholding
        if img is None:
            print("Error: Unable to read the image.")
            return
        
        _, binaryImage = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

        cropped_images = self.modified_horizontal_projection(binaryImage)

        letter_images = self.extract_letters(cropped_images)

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        for idx, letter_img in enumerate(letter_images):
            image_output_path = os.path.join(outputPath, f"{idx}.png")
            cv2.imwrite(image_output_path, letter_img)
        
        print(f"Saved {len(letter_images)} letter images.")



#NOTE: Connected Components code
    def process_images_CC(Self, input_folder: str, output_folder: str):
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        stop = 0
        # Loop through all image files in the input folder
        for filename in os.listdir(input_folder):
            # Skip files that start with "hard"
            if filename.lower().startswith("hard"):
                print(f"[INFO] Skipping {filename} (excluded by filter)")
                continue
               
            # Process only image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):  
                image_path = os.path.join(input_folder, filename)
                Self.process_single_image_CC(image_path, output_folder)
            
            stop += 1
            if stop == 5:
                break

    def process_single_image_CC(Self, image_path: str, output_folder: str):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Unable to load {image_path}")
            return
        
        img_height, img_width = img.shape[:2]

        # Convert to grayscale if necessary
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        output = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
        numLabels, labels, stats, centroids = output

        afterFiltering = 0
        for i in range(1, numLabels):  # Skipping background (0)
            x, y, w, h, area = stats[i]
            cX, cY = centroids[i]
            
            # Filtering out small components
            if w > Self.minimumWidthForAsomtavruli and h > Self.minimumHeightForAsomtavruli:
                timestamp = int(time.time() * 1000)  # Unique identifier
                unique_name = f"component_{timestamp}_{i}"
                afterFiltering += 1
                # Extract the component
                component = img[y:y+h, x:x+w]
                image_output_path = os.path.join(output_folder, f"{unique_name}.png")
                txt_output_path = os.path.join(output_folder, f"{unique_name}.txt")

                # Save component image
                cv2.imwrite(image_output_path, component)

                # Save component stats
                with open(txt_output_path, 'w') as f:
                    f.write(f"Location: {os.path.basename(image_path)}\n")
                    f.write(f"Component: {unique_name}.png\n")
                    f.write(f"centerX: {x / img_width}\n")
                    f.write(f"centerY: {y / img_height}\n")
                    f.write(f"width: {w / img_width}\n")
                    f.write(f"height: {h / img_height}\n")
                    f.write(f"Area: {area / (img_width * img_height)}\n")
                    f.write(f"CentroidX: {(cX/ img_width):.4f}\n")
                    f.write(f"CentroidY: {(cY / img_height):.4f}\n")
                    # f.write(f"Component: {i}\n")
                    # f.write(f"Position: (x={x}, y={y})\n")
                    # f.write(f"Size: (width={w}, height={h})\n")
                    # f.write(f"Area: {area}\n")
                    # f.write(f"Centroid: (cX={cX:.2f}, cY={cY:.2f})\n")

                print(f"[INFO] Saved {unique_name}.png and {unique_name}.txt")

        print(f"[INFO] Processing completed for {image_path}")
        print(f"amount of components before filtering: {numLabels}")
        print(f"amount of components after filtering: {afterFiltering}")
            
    # NOTE: this does really well on the normal pictures, however very poorly on the hard ones.
    def showFilteredConnectedComponents(Self, path:str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        copy = img
        _, new = cv2.threshold(copy, 128, 255, cv2.THRESH_BINARY_INV)
        output = cv2.connectedComponentsWithStats(new, 4, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        numLabels = output[0]
        # The second cell is the label matrix (actual connected components found)
        labels = output[1]
        # The third cell is the stat matrix (x,y,height,width of said components)
        stats = output[2]
        # The fourth cell is the centroid matrix (collection of points that are the centres of said components)
        centroids = output[3]
        
        mask = np.zeros_like(labels, dtype="uint8")
        
        # loop over the number of unique connected component labels
        for i in range(1, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(i + 1, numLabels)
         # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format(i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            
            # ensure the width, height, and area are all neither too small
	        # nor too big
            keepWidth = w > 5 
            keepHeight = h > 10
            # keepArea = area > 500 and area < 1500
	        # ensure the connected component we are examining passes all
	        # three tests
            if all((keepWidth, keepHeight)):
	        	# construct a mask for the current connected component and
	        	# then take the bitwise OR with the mask
                print("[INFO] keeping connected component '{}'".format(i))
                componentMask = (labels == i).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)
                
            # show the original input image and the mask for the license plate
            # characters
                cv2.imshow("Image", new)
                cv2.imshow("Characters", mask)
                cv2.waitKey(0)

    def showAllConnectedComponents(Self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        copy = img
        _, new = cv2.threshold(copy, 128, 255, cv2.THRESH_BINARY_INV)
        output = cv2.connectedComponentsWithStats(new, 4, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        numLabels = output[0]
        # The second cell is the label matrix (actual connected components found)
        labels = output[1]
        # The third cell is the stat matrix (x,y,height,width of said components)
        stats = output[2]
        # The fourth cell is the centroid matrix (collection of points that are the centres of said components)
        centroids = output[3]
        
        # loop over the number of unique connected component labels
        for i in range(0, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(i + 1, numLabels)
         # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format(i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            
            
            output = img.copy()# clone our original image (so we can draw on it) 
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 3)# a bounding box surrounding the connected component along with
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)# a circle corresponding to the centroid
            
            # construct a mask for the current connected component by
	        # finding a pixels in the labels array that have the current
	        # connected component ID 
            componentMask = (labels == i).astype("uint8") * 255
	        # show our output image and connected component mask
            cv2.imshow("Output", output)
            # cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)
            
def slice_letters_and_update_metadata(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Average dimensions
    COMPONENT_W, COMPONENT_H = 51.08, 25.25
    HARD_W, HARD_H = 206.54, 119.15

    for file in os.listdir(input_dir):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(input_dir, file)
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        if not os.path.exists(txt_path):
            print(f"Missing metadata for {file}, skipping.")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {file}, skipping.")
            continue

        height, width = img.shape
        base_name = os.path.splitext(file)[0]

        # Determine type and average dimensions
        if file.startswith("component"):
            avg_w, avg_h = COMPONENT_W, COMPONENT_H
        elif file.startswith("hard"):
            avg_w, avg_h = HARD_W, HARD_H
        else:
            print(f"Unknown type for {file}, skipping.")
            continue

        # Compute number of splits needed in each direction
        x_slices = max(1, math.ceil(width / avg_w))
        y_slices = max(1, math.ceil(height / avg_h))
        slice_w = width // x_slices
        slice_h = height // y_slices

        # Read original metadata lines (we keep Location line)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        location_line = next((line for line in lines if line.startswith("Location:")), "Location:Unknown\n")

        # Save each slice
        counter = 1
        for y in range(y_slices):
            for x in range(x_slices):
                x_start = x * slice_w
                y_start = y * slice_h
                x_end = width if x == x_slices - 1 else (x + 1) * slice_w
                y_end = height if y == y_slices - 1 else (y + 1) * slice_h

                sub_img = img[y_start:y_end, x_start:x_end]
                sub_h, sub_w = sub_img.shape

                out_img_name = f"{base_name}_{counter}.png"
                out_txt_name = f"{base_name}_{counter}.txt"
                out_img_path = os.path.join(output_dir, out_img_name)
                out_txt_path = os.path.join(output_dir, out_txt_name)

                # Save image
                cv2.imwrite(out_img_path, sub_img)

                # Compute normalized metadata values
                centerX = 0.5
                centerY = 0.5
                width_norm = sub_w / width
                height_norm = sub_h / height
                area_norm = (sub_w * sub_h) / (width * height)
                centroidX = (x_start + sub_w / 2) / width
                centroidY = (y_start + sub_h / 2) / height

                # Write metadata
                with open(out_txt_path, 'w', encoding='utf-8') as f:
                    f.write(location_line)
                    f.write(f"Component: {out_img_name}\n")
                    f.write(f"centerX: {centerX:.4f}\n")
                    f.write(f"centerY: {centerY:.4f}\n")
                    f.write(f"width: {width_norm:.4f}\n")
                    f.write(f"height: {height_norm:.4f}\n")
                    f.write(f"Area: {area_norm:.4f}\n")
                    f.write(f"CentroidX: {centroidX:.4f}\n")
                    f.write(f"CentroidY: {centroidY:.4f}\n")

                counter += 1

    return "✔ Done slicing images and generating updated metadata."



            

obj = SegmentationManager("")
obj.process_images_CC( "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Thresholded images",
                            "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Segmented Images")
# obj.showFilteredConnectedComponents("/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Thresholded images/Asomtavruli Data Output/easy1_Gaussian-Closing.png")

# obj.segment_PP("/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted/Conjoined/hard7_Mean-Medain_contour_02febd78.png",
#                             "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Segmented Images")

detector = ContourDetector(min_area=1500, output_dir="/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Segmented Images")  # Set a minimum area for filtering
# processed_image, contours = detector.process_image("/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Thresholded images/Asomtavruli Data Output/hard1_Gaussian-Closing.png", save_contours=True)

detector.process_folder("/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Thresholded images/")  # Replace with the actual pat

# detector.process_image("/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted/Conjoined/hard7_Mean-Medain_contour_02febd78.png", True)
# slice_letters_and_update_metadata("/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted/Conjoined",
#                     "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Segmented Images")
