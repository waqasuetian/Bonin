# import streamlit as st
# import os
# import tempfile
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import pydicom
# import shutil
# import zipfile
# import cv2
# import re
# from concurrent.futures import ThreadPoolExecutor
# from zipfile import ZipFile

# # ------------ Session state setup ----------------
# if 'annotation_dict' not in st.session_state:
#     st.session_state.annotation_dict = {}

# if 'single_dicom_ready' not in st.session_state:
#     st.session_state.single_dicom_ready = False

# # ------------ Load model ----------------
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model(
#         r"D:\PhD Research\Experiments\Tooth_Map\Model\Dental_Segmentation-main\model100.h5",
#         compile=False
#     )
#     return model

# model = load_model()

# # ------------ DICOM to PNG ----------------
# def dicom_to_png(dicom_file, size=(256, 256)):
#     print("1st")
#     dicom_data = pydicom.dcmread(dicom_file)
#     pixel_array = dicom_data.pixel_array.astype(np.float32)
#     pixel_array -= np.min(pixel_array)
#     pixel_array /= np.max(pixel_array)
#     pixel_array *= 255.0
#     pixel_array = pixel_array.astype(np.uint8)
#     img = Image.fromarray(pixel_array).convert("L")
#     img_resized = img.resize(size)

#     img_arr = np.array(img_resized, dtype=np.float32) / 255.0
#     img_arr = np.expand_dims(img_arr, axis=-1)
#     img_arr = np.expand_dims(img_arr, axis=0)

#     return img_arr, img_resized

# # ------------ Predict mask ----------------
# def predict_and_save_mask(model, preprocessed_input, output_path):
#     print("2nt")
#     prediction = model.predict(preprocessed_input)[0]
#     mask = prediction[:, :, 0] if prediction.shape[-1] == 1 else prediction
#     mask_img = Image.fromarray((mask * 255).astype(np.uint8))
#     mask_img.save(output_path)
#     return mask

# # ------------ Padding ----------------
# def pad_mask_image(mask_path, pad_top=9, pad_left=12):
#     print("3rd")
#     img = Image.open(mask_path)
#     padded = Image.new("L", (img.width + pad_left, img.height + pad_top), 0)
#     padded.paste(img, (pad_left, pad_top))
#     padded_path = mask_path.replace(".png", "_padded.png")
#     padded.save(padded_path)
#     return padded_path

# # ------------ Annotation Dictionary ----------------
# # ----------------------------
# def build_annotation_dictionary(image_folder, pad_top=9, pad_left=15):
#     print("4th")
#     color_ranges = {
#         "Root": ([35, 50, 50], [85, 255, 255]),
#         "Nerve": ([90, 50, 50], [130, 255, 255]),
#         "Enamel": ([10, 50, 50], [25, 255, 255])
#     }

#     def process_image(filename):
#         image_path = os.path.join(image_folder, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             return filename, {}

#         padded_image = cv2.copyMakeBorder(image, pad_top, 0, pad_left, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         hsv = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)

#         color_coordinates = {}
#         for label, (lower, upper) in color_ranges.items():
#             mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
#             coords = np.column_stack(np.where(mask > 0))
#             coords = [(int(x), int(y)) for y, x in coords]
#             color_coordinates[label] = coords
#         return filename, color_coordinates

#     all_color_coordinates = {}
#     with ThreadPoolExecutor() as executor:
#         results = executor.map(process_image, os.listdir(image_folder))
#         for filename, coords in results:
#             all_color_coordinates[filename] = coords
#     print(all_color_coordinates)        
#     return all_color_coordinates

# import os
# import re
# import numpy as np
# import pydicom
# from PIL import Image, ImageDraw
# from zipfile import ZipFile

# # ------------ Annotate series ----------------
# def annotate_dicom_series(annotation_dict, dicom_dir, output_dir, zip_output_path=None):
#     print("5tt")

#     # RGB colors for annotation
#     colors = {
#         'Root': (0, 255, 0),      # Green
#         'Nerve': (0, 0, 255),     # Blue
#         'Enamel': (255, 165, 0)   # Orange
#     }

#     def apply_ct_window(img, window):
#         R = (img - window[1] + 0.5 * window[0]) / window[0]
#         R[R < 0] = 0
#         R[R > 1] = 1
#         return R

#     os.makedirs(output_dir, exist_ok=True)
#     saved_files = []

#     for filename in sorted(os.listdir(dicom_dir)):
#         if not filename.endswith(".dcm"):
#             continue

#         base = os.path.splitext(filename)[0]
#         match = re.search(r'\d+', base)
#         if match:
#             num = match.group()
#             mask_key = f"mask_{num}_padded.png"
#         else:
#             mask_key = f"{base}_padded.png"

#         if mask_key not in annotation_dict:
#             print(f"⚠️ No annotation for {mask_key}")
#             continue

#         dicom_path = os.path.join(dicom_dir, filename)
#         output_path = os.path.join(output_dir, f"{base}_annot.dcm")

#         ds = pydicom.dcmread(dicom_path)
#         img = ds.pixel_array.astype(float)
#         img = img * ds.RescaleSlope + ds.RescaleIntercept

#         display_img = apply_ct_window(img, [400, 50])
#         rgb_img = Image.fromarray((255 * display_img).astype('uint8')).convert('RGB')

#         draw = ImageDraw.Draw(rgb_img)
#         for label, coords in annotation_dict[mask_key].items():
#             if label not in colors:
#                 continue
#             for y, x in coords:
#                 draw.point((x, y), fill=colors[label])

#         rgb_array = np.array(rgb_img)

#         ds.PhotometricInterpretation = 'RGB'
#         ds.SamplesPerPixel = 3
#         ds.BitsAllocated = 16
#         ds.BitsStored = 16
#         ds.HighBit = 7
#         ds.Rows, ds.Columns = rgb_array.shape[:2]
#         ds.PlanarConfiguration = 0
#         ds.PixelRepresentation = 0
#         ds.PixelData = rgb_array.tobytes()

#         ds.save_as(output_path)
#         saved_files.append(output_path)
#         print(f"✅ Annotated and saved: {output_path}")

#     # Write zip if specified
#     if zip_output_path:
#         with ZipFile(zip_output_path, 'w') as zipf:
#             for file in saved_files:
#                 arcname = os.path.basename(file)
#                 zipf.write(file, arcname)
#         print(f"✅ Zipped {len(saved_files)} files to {zip_output_path}")
#         if not saved_files:
#             print("⚠️ No DICOMs were annotated. Empty ZIP created.")


# # ------------ Streamlit UI ----------------
# st.title("🦷 Dental DICOM Annotator - Case 1 & 2")

# tab1, tab2 = st.tabs(["📁 Case 1: Full DICOM Series", "🧪 Case 2: Single + Annotate All"])

# # ---------------- CASE 1 ----------------
# with tab1:
#     uploaded_files = st.file_uploader("Upload full DICOM series (export1.dcm ... exportN.dcm)", type=["dcm"], accept_multiple_files=True)

#     if uploaded_files:
#         with tempfile.TemporaryDirectory() as tmpdir:
#             dicom_dir = os.path.join(tmpdir, "dicoms")
#             os.makedirs(dicom_dir, exist_ok=True)

#             for file in uploaded_files:
#                 out_path = os.path.join(dicom_dir, file.name)
#                 with open(out_path, "wb") as f:
#                     f.write(file.read())

#             png_dir = os.path.join(tmpdir, "pngs")
#             mask_dir = os.path.join(tmpdir, "masks")
#             os.makedirs(png_dir, exist_ok=True)
#             os.makedirs(mask_dir, exist_ok=True)

#             for f in sorted(os.listdir(dicom_dir)):
#                 input_path = os.path.join(dicom_dir, f)
#                 idx = f.replace("export", "").replace(".dcm", "")
#                 img_arr, png_img = dicom_to_png(input_path)
#                 Image.fromarray((img_arr[0, :, :, 0] * 255).astype(np.uint8)).save(os.path.join(png_dir, f"slice_{idx}.png"))

#                 pred_path = os.path.join(mask_dir, f"mask_{idx}.png")
#                 _ = predict_and_save_mask(model, img_arr, pred_path)
#                 pad_mask_image(pred_path)

#             st.success("✅ PNGs, Masks and Padded Masks generated.")

#             st.info("🔍 Building annotation dictionary...")
#             annotation_dict = build_annotation_dictionary(mask_dir)
#             st.session_state.annotation_dict = annotation_dict
#             st.success("✅ Annotation dictionary ready.")
#             print(annotation_dict)

#             st.info("💾 Annotating original DICOMs...")
#             zip_path = os.path.join(tmpdir, "annotated_dicoms.zip")
#             output_folder = os.path.join(tmpdir, "annotated")
#             annotate_dicom_series(annotation_dict, dicom_dir, output_folder, zip_path)

#             if os.path.exists(zip_path):
#                 with open(zip_path, "rb") as f:
#                     st.download_button("⬇ Download Annotated Series", f, file_name="annotated_dicoms.zip")
#             else:
#                 st.error("❌ ZIP file was not created. No DICOMs were annotated.")


# # import os
# # import tempfile
# # import shutil
# # import zipfile
# # import streamlit as st
# # import numpy as np
# # import tensorflow as tf
# # from PIL import Image
# # import pydicom

# # import cv2
# # from concurrent.futures import ThreadPoolExecutor

# # # ----------------------------
# # @st.cache_resource
# # def load_model():
# #     model = tf.keras.models.load_model(
# #         r"D:\PhD Research\Experiments\Tooth_Map\Model\Dental_Segmentation-main\model100.h5",
# #         compile=False
# #     )
# #     return model

# # # ----------------------------
# # def dicom_to_png(dicom_file, size=(256, 256)):
# #     dicom_data = pydicom.dcmread(dicom_file)
# #     pixel_array = dicom_data.pixel_array.astype(np.float32)

# #     pixel_array -= np.min(pixel_array)
# #     pixel_array /= np.max(pixel_array)
# #     pixel_array *= 255.0
# #     pixel_array = pixel_array.astype(np.uint8)

# #     img = Image.fromarray(pixel_array).convert("L")
# #     img_resized = img.resize(size)

# #     img_arr = np.array(img_resized, dtype=np.float32) / 255.0
# #     img_arr = np.expand_dims(img_arr, axis=-1)
# #     img_arr = np.expand_dims(img_arr, axis=0)

# #     return img_arr, img_resized

# # # ----------------------------
# # def predict_and_save_mask(model, preprocessed_input, output_path):
# #     prediction = model.predict(preprocessed_input)[0]
# #     mask = prediction[:, :, 0] if prediction.shape[-1] == 1 else prediction
# #     mask_img = Image.fromarray((mask * 255).astype(np.uint8))
# #     mask_img.save(output_path)
# #     return mask

# # # ----------------------------
# # def build_annotation_dictionary(image_folder, pad_top=9, pad_left=15):
# #     color_ranges = {
# #         "Root": ([35, 50, 50], [85, 255, 255]),
# #         "Nerve": ([90, 50, 50], [130, 255, 255]),
# #         "Enamel": ([10, 50, 50], [25, 255, 255])
# #     }

# #     def process_image(filename):
# #         image_path = os.path.join(image_folder, filename)
# #         image = cv2.imread(image_path)
# #         if image is None:
# #             return filename, {}

# #         padded_image = cv2.copyMakeBorder(image, pad_top, 0, pad_left, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
# #         hsv = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)

# #         color_coordinates = {}
# #         for label, (lower, upper) in color_ranges.items():
# #             mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
# #             coords = np.column_stack(np.where(mask > 0))
# #             coords = [(int(x), int(y)) for y, x in coords]
# #             color_coordinates[label] = coords
# #         return filename, color_coordinates

# #     all_color_coordinates = {}
# #     with ThreadPoolExecutor() as executor:
# #         results = executor.map(process_image, os.listdir(image_folder))
# #         for filename, coords in results:
# #             all_color_coordinates[filename] = coords

# #     return all_color_coordinates

# # # ----------------------------
# # def annotate_dicom_series(annotation_dict, dicom_dir, output_dir, zip_output_path=None):
# #     gray_values = {
# #         'Root':   500,
# #         'Nerve':  15000,
# #         'Enamel': 25000,
# #     }

# #     os.makedirs(output_dir, exist_ok=True)
# #     saved_files = []

# #     for i in range(1, 179):
# #         mask_key = f"mask_{i}.png"
# #         filename = f"export{i}.dcm"
# #         slice_id = os.path.splitext(filename)[0]

# #         dicom_path = os.path.join(dicom_dir, filename)
# #         out_path = os.path.join(output_dir, f"{slice_id}_annot.dcm")

# #         if mask_key not in annotation_dict:
# #             continue

# #         ds = pydicom.dcmread(dicom_path)
# #         arr = ds.pixel_array.copy().astype(np.int32)
# #         h, w = arr.shape

# #         for label, pts in annotation_dict[mask_key].items():
# #             shade = gray_values.get(label)
# #             if shade is None:
# #                 continue
# #             for y, x in pts:
# #                 if 0 <= y < w and 0 <= x < h:
# #                     arr[x, y] = shade

# #         orig_dtype = ds.pixel_array.dtype
# #         arr = np.clip(arr, np.iinfo(orig_dtype).min, np.iinfo(orig_dtype).max).astype(orig_dtype)
# #         ds.PixelData = arr.tobytes()
# #         ds.save_as(out_path)
# #         saved_files.append(out_path)

# #     if zip_output_path:
# #         with zipfile.ZipFile(zip_output_path, 'w') as zipf:
# #             for file in saved_files:
# #                 arcname = os.path.basename(file)
# #                 zipf.write(file, arcname)
# #         return zip_output_path
# #     return None

# # # ----------------------------
# # # Streamlit UI
# # st.title("Dental AI Annotation Tool")

# # model = load_model()

# # uploaded_dicom = st.file_uploader("Upload a DICOM slice", type=["dcm"])

# # if uploaded_dicom:
# #     with tempfile.TemporaryDirectory() as tmpdir:
# #         dicom_path = os.path.join(tmpdir, "input.dcm")
# #         with open(dicom_path, "wb") as f:
# #             f.write(uploaded_dicom.read())

# #         input_arr, dicom_img = dicom_to_png(dicom_path)
# #         st.image(dicom_img, caption="Converted DICOM Slice", use_column_width=True)

# #         mask_path = os.path.join(tmpdir, "mask_1.png")
# #         _ = predict_and_save_mask(model, input_arr, mask_path)
# #         st.image(mask_path, caption="Predicted Mask", use_column_width=True)

# #         # Create mask folder for annotation
# #         mask_folder = os.path.join(tmpdir, "masks")
# #         os.makedirs(mask_folder, exist_ok=True)
# #         shutil.move(mask_path, os.path.join(mask_folder, "mask_1.png"))

# #         annotation_dict = build_annotation_dictionary(mask_folder)

# #         dicom_dir = os.path.dirname(dicom_path)
# #         output_dir = os.path.join(tmpdir, "annotated_dicoms")
# #         zip_output_path = os.path.join(tmpdir, "annotated_dicoms.zip")

# #         zip_path = annotate_dicom_series(
# #             annotation_dict, dicom_dir, output_dir, zip_output_path=zip_output_path
# #         )

# #         if zip_path:
# #             with open(zip_path, "rb") as f:
# #                 st.download_button(
# #                     label="Download Annotated DICOMs (ZIP)",
# #                     data=f,
# #                     file_name="annotated_dicoms.zip",
# #                     mime="application/zip"
# #                 )

import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import tensorflow as tf
import pydicom
import shutil
from zipfile import ZipFile
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
import re

# # ------------ Session state setup ----------------
# if 'annotation_dict' not in st.session_state:
#     st.session_state.annotation_dict = {}

# if 'single_dicom_ready' not in st.session_state:
#     st.session_state.single_dicom_ready = False

# # ------------ Load model ----------------
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model(
#         r"model100.h5",
#         compile=False
#     )
#     return model

# model = load_model()

# # ------------ DICOM to PNG ----------------
# def dicom_to_png(dicom_file, size=(256, 256)):
#     dicom_data = pydicom.dcmread(dicom_file)
#     pixel_array = dicom_data.pixel_array.astype(np.float32)
#     pixel_array -= np.min(pixel_array)
#     pixel_array /= np.max(pixel_array)
#     pixel_array *= 255.0
#     pixel_array = pixel_array.astype(np.uint8)
#     img = Image.fromarray(pixel_array).convert("L")
#     img_resized = img.resize(size)

#     img_arr = np.array(img_resized, dtype=np.float32) / 255.0
#     img_arr = np.expand_dims(img_arr, axis=-1)
#     img_arr = np.expand_dims(img_arr, axis=0)

#     return img_arr, img_resized

# # ------------ Predict mask ----------------
# def predict_and_save_mask(model, preprocessed_input, output_path):
#     prediction = model.predict(preprocessed_input)[0]
#     mask = prediction[:, :, 0] if prediction.shape[-1] == 1 else prediction
#     mask_img = Image.fromarray((mask * 255).astype(np.uint8))
#     mask_img.save(output_path)
#     return mask

# # ------------ Padding ----------------
# def pad_mask_image(mask_path, pad_top=9, pad_left=12):
#     img = Image.open(mask_path)
#     padded = Image.new("L", (img.width + pad_left, img.height + pad_top), 0)
#     padded.paste(img, (pad_left, pad_top))
#     padded_path = mask_path.replace(".png", "_padded.png")
#     padded.save(padded_path)
#     return padded_path

#------------ Annotation Dictionary ----------------


# def build_annotation_dictionary(image_folder, pad_top=9, pad_left=15):
#     color_ranges = {
#         "Root": ([35, 50, 50], [85, 255, 255]),
#         "Nerve": ([90, 50, 50], [130, 255, 255]),
#         "Enamel": ([10, 50, 50], [25, 255, 255])
#     }

#     def process_image(filename):
#         image_path = os.path.join(image_folder, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             return filename, {}
#         padded_image = cv2.copyMakeBorder(image, pad_top, 0, pad_left, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         hsv = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)
#         color_coordinates = {}
#         for label, (lower, upper) in color_ranges.items():
#             mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
#             coords = np.column_stack(np.where(mask > 0))
#             coords = [(int(x), int(y)) for y, x in coords]
#             color_coordinates[label] = coords
#         return filename, color_coordinates

#     all_coords = {}
#     with ThreadPoolExecutor() as executor:
#         for filename, coords in executor.map(process_image, os.listdir(image_folder)):
#             all_coords[filename] = coords
#     return all_coords

# def annotate_dicom_series(annotation_dict, dicom_dir, output_dir, zip_output_path=None):
   

#     os.makedirs(output_dir, exist_ok=True)
#     saved_files = []

#     colors = {
#         "Root": (0, 255, 0),      # Green
#         "Nerve": (0, 0, 255),     # Red
#         "Enamel": (255, 255, 255) # White
#     }

#     def apply_ct_window(img, window):
#         R = (img - window[1] + 0.5 * window[0]) / window[0]
#         R[R < 0] = 0
#         R[R > 1] = 1
#         return R

#     for filename in sorted(os.listdir(dicom_dir)):
#         if not filename.endswith(".dcm"):
#             continue

#         base = os.path.splitext(filename)[0]
#         match = re.search(r'\d+', base)
#         if match:
#             number = match.group()
#             mask_key = f"mask_{number}_padded.png"
#         else:
#             print(f"⚠️ Could not extract number from filename: {filename}")
#             continue

#         if mask_key not in annotation_dict:
#             print(f"⚠️ No annotation for {mask_key}")
#             continue

#         dicom_path = os.path.join(dicom_dir, filename)
#         out_path = os.path.join(output_dir, base + "_annot.dcm")

#         ds = pydicom.dcmread(dicom_path)
#         img = ds.pixel_array.astype(float)
#         img = img * ds.RescaleSlope + ds.RescaleIntercept
#         display_img = apply_ct_window(img, [400, 50])

#         rgb_img = Image.fromarray((255 * display_img).astype('uint8')).convert('RGB')
#         draw = ImageDraw.Draw(rgb_img)

#         mask_coords = annotation_dict[mask_key]
#         for label, coords in mask_coords.items():
#             color = colors[label]
#             for y, x in coords:
#                 draw.point((x, y), fill=color)

#         rgb_array = np.array(rgb_img)

#         # Update DICOM metadata
#         ds.PhotometricInterpretation = 'RGB'
#         ds.SamplesPerPixel = 3
#         ds.BitsAllocated = 8
#         ds.BitsStored = 8
#         ds.HighBit = 7
#         ds.PixelData = rgb_array.tobytes()
#         ds.Rows, ds.Columns = rgb_array.shape[:2]
#         ds.PlanarConfiguration = 0
#         ds.is_little_endian = True

#         ds.save_as(out_path)
#         saved_files.append(out_path)

#     if zip_output_path:
#         with ZipFile(zip_output_path, "w") as zipf:
#             for f in saved_files:
#                 zipf.write(f, os.path.basename(f))

#2nd

# def build_annotation_dictionary(image_folder, pad_top=9, pad_left=15):
#     color_ranges = {
#         "Root": ([35, 50, 50], [85, 255, 255]),
#         "Nerve": ([90, 50, 50], [130, 255, 255]),
#         "Enamel": ([10, 50, 50], [25, 255, 255])
#     }

#     def process_image(filename):
#         image_path = os.path.join(image_folder, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             return filename, {}
            
#         # Apply padding to match DICOM dimensions
#         padded_image = cv2.copyMakeBorder(image, pad_top, 0, pad_left, 0, 
#                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         hsv = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)
        
#         color_coordinates = {}
#         for label, (lower, upper) in color_ranges.items():
#             mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), 
#                              np.array(upper, dtype=np.uint8))
#             coords = np.column_stack(np.where(mask > 0))
#             # Convert to (x, y) format and store as integers
#             color_coordinates[label] = [(int(x), int(y)) for y, x in coords]
            
#         return filename, color_coordinates

#     all_coords = {}
#     with ThreadPoolExecutor() as executor:
#         results = executor.map(process_image, os.listdir(image_folder))
#         for filename, coords in results:
#             all_coords[filename] = coords
            
#     return all_coords

# def annotate_dicom_series(annotation_dict, dicom_dir, output_dir, zip_output_path=None):
#     os.makedirs(output_dir, exist_ok=True)
#     saved_files = []

#     # High-contrast grayscale values
#     gray_values = {
#         'Root': 500,     # Dark gray
#         'Nerve': 15000,  # Medium gray
#         'Enamel': 25000  # Bright white
#     }

#     # Padding compensation (should match build_annotation_dictionary)
#     pad_left = 15
#     pad_top = 9

#     for filename in sorted(os.listdir(dicom_dir)):
#         if not filename.endswith(".dcm"):
#             continue

#         base = os.path.splitext(filename)[0]
#         if (match := re.search(r'\d+', base)):
#             mask_key = f"mask_{match.group()}_padded.png"
#         else:
#             st.warning(f"Couldn't extract number from {filename}")
#             continue

#         if mask_key not in annotation_dict:
#             st.warning(f"No annotations for {mask_key}")
#             continue

#         # Read and prepare DICOM
#         dicom_path = os.path.join(dicom_dir, filename)
#         ds = pydicom.dcmread(dicom_path)
#         arr = ds.pixel_array.copy().astype(np.int32)
#         h, w = arr.shape

#         # Apply annotations with padding compensation
#         for label, coords in annotation_dict[mask_key].items():
#             if (shade := gray_values.get(label)) is None:
#                 continue
                
#             for x, y in coords:
#                 # Adjust for mask padding and ensure within bounds
#                 adj_x = x - pad_left
#                 adj_y = y - pad_top
#                 if 0 <= adj_x < w and 0 <= adj_y < h:
#                     arr[adj_y, adj_x] = shade

#         # Maintain original data type constraints
#         arr = np.clip(arr, np.iinfo(ds.pixel_array.dtype).min,
#                       np.iinfo(ds.pixel_array.dtype).max)
#         ds.PixelData = arr.astype(ds.pixel_array.dtype).tobytes()

#         # Save annotated DICOM
#         out_path = os.path.join(output_dir, f"{base}_annot.dcm")
#         ds.save_as(out_path)
#         saved_files.append(out_path)

#     # Create ZIP archive if requested
#     if zip_output_path:
#         with ZipFile(zip_output_path, 'w') as zipf:
#             for f in saved_files:
#                 zipf.write(f, os.path.basename(f))

#     return saved_files
# --------------------------------------------------

# def build_annotation_dictionary(image_folder, pad_top=9, pad_left=15):
#     color_ranges = {
#         "Root": ([35, 50, 50], [85, 255, 255]),
#         "Nerve": ([90, 50, 50], [130, 255, 255]),
#         "Enamel": ([10, 50, 50], [25, 255, 255])
#     }

#     def process_image(filename):
#         image_path = os.path.join(image_folder, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             return filename, {}
            
#         # Apply padding to match DICOM dimensions
#         padded_image = cv2.copyMakeBorder(image, pad_top, 0, pad_left, 0, 
#                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         hsv = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)
        
#         color_coordinates = {}
#         for label, (lower, upper) in color_ranges.items():
#             mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), 
#                              np.array(upper, dtype=np.uint8))
#             coords = np.column_stack(np.where(mask > 0))
#             # Convert to (x, y) format and store as integers
#             color_coordinates[label] = [(int(x), int(y)) for y, x in coords]
            
#         return filename, color_coordinates

#     all_coords = {}
#     with ThreadPoolExecutor() as executor:
#         results = executor.map(process_image, os.listdir(image_folder))
#         for filename, coords in results:
#             all_coords[filename] = coords
            
#     return all_coords

# def annotate_dicom_series(annotation_dict, dicom_dir, output_dir, feature, zip_output_path=None):
#     os.makedirs(output_dir, exist_ok=True)
#     saved_files = []
#     print(feature)

#     # High-contrast grayscale values
#     gray_values = {
#         'Root': 500,     # Dark gray
#         'Nerve': 15000,  # Medium gray
#         'Enamel': 25000  # Bright white
#     }
    
#     if feature == "ALL":
#         gray_values = {     
#         'Root': 25000,     # Dark gray
#         'Nerve': 25000,  # Medium gray
#         'Enamel': 25000  # Bright white
#     }
       
#     if feature == "Nerve":
#         gray_values = {     
#         'Root': 500,     # Dark gray
#         'Nerve': 25000,  # Medium gray
#         'Enamel': 1500  # Bright white
#     }
#     if feature == "Enamel":
#         gray_values = {     
#         'Root': 500,     # Dark gray
#         'Nerve': 1500,  # Medium gray
#         'Enamel': 25000  # Bright white
#     }
#     if feature == "Root":
#         gray_values = {     
#         'Root': 25000,     # Dark gray
#         'Nerve': 1500,  # Medium gray
#         'Enamel': 500  # Bright white
#     }
    # # Set selected feature to bright value
    # if feature in gray_values:
    #     print("ata be ha ya ni")
    #     gray_values[feature] = 25000
    
    # Padding compensation (should match build_annotation_dictionary)
    # pad_left = 15
    # pad_top = 9

    # for filename in sorted(os.listdir(dicom_dir)):
    #     if not filename.endswith(".dcm"):
    #         continue

    #     base = os.path.splitext(filename)[0]
    #     if (match := re.search(r'\d+', base)):
    #         mask_key = f"mask_{match.group()}_padded.png"
    #     else:
    #         st.warning(f"Couldn't extract number from {filename}")
    #         continue

    #     if mask_key not in annotation_dict:
    #         st.warning(f"No annotations for {mask_key}")
    #         continue

    #     # Read and prepare DICOM
    #     dicom_path = os.path.join(dicom_dir, filename)
    #     ds = pydicom.dcmread(dicom_path)
    #     arr = ds.pixel_array.copy().astype(np.int32)
    #     h, w = arr.shape

    #     # Apply annotations with padding compensation
    #     for label, coords in annotation_dict[mask_key].items():
    #         if (shade := gray_values.get(label)) is None:
    #             continue
                
    #         for x, y in coords:
    #             # Adjust for mask padding and ensure within bounds
    #             adj_x = x - pad_left
    #             adj_y = y - pad_top
    #             if 0 <= adj_x < w and 0 <= adj_y < h:
    #                 arr[adj_y, adj_x] = shade

    #     # Maintain original data type constraints
    #     arr = np.clip(arr, np.iinfo(ds.pixel_array.dtype).min,
    #                   np.iinfo(ds.pixel_array.dtype).max)
    #     ds.PixelData = arr.astype(ds.pixel_array.dtype).tobytes()

    #     # Save annotated DICOM
    #     out_path = os.path.join(output_dir, f"{base}_annot.dcm")
    #     ds.save_as(out_path)
    #     saved_files.append(out_path)

    # # Create ZIP archive if requested
    # if zip_output_path:
    #     with ZipFile(zip_output_path, 'w') as zipf:
    #         for f in saved_files:
    #             zipf.write(f, os.path.basename(f))

    # return saved_files

# ------------ Session state setup ----------------
if 'annotation_dict' not in st.session_state:
    st.session_state.annotation_dict = {}
if 'single_dicom_ready' not in st.session_state:
    st.session_state.single_dicom_ready = False
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# ------------ Load model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"model100.h5", compile=False)

model = load_model()

# ------------ DICOM to PNG with enhanced normalization ----------------
def dicom_to_png(dicom_file, size=(256, 256)):
    dicom_data = pydicom.dcmread(dicom_file)
    pixel_array = dicom_data.pixel_array.astype(np.float32)
    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    pixel_array = (pixel_array * 255).astype(np.uint8)
    
    img = Image.fromarray(pixel_array).convert("L")
    img_resized = img.resize(size)
    
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    return img_arr, img_resized

# ------------ Mask prediction ----------------
def predict_and_save_mask(model, preprocessed_input, output_path):
    prediction = model.predict(preprocessed_input)[0]
    mask = prediction[:, :, 0] if prediction.shape[-1] == 1 else prediction
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(output_path)
    return mask

# ------------ Annotation dictionary builder ----------------
def build_annotation_dictionary(image_folder):
    color_ranges = {
        "Root": ([35, 50, 50], [85, 255, 255]),
        "Nerve": ([90, 50, 50], [130, 255, 255]),
        "Enamel": ([10, 50, 50], [25, 255, 255])
    }

    def process_image(filename):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            return filename, {}

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_coordinates = {}
        
        for label, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            coords = np.column_stack(np.where(mask > 0))
            color_coordinates[label] = [(int(x), int(y)) for y, x in coords]

        return filename, color_coordinates

    all_coords = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image, os.listdir(image_folder))
        for filename, coords in results:
            all_coords[filename] = coords

    return all_coords

# ------------ Core DICOM annotation function ----------------
def annotate_dicom_series(annotation_dict, dicom_dir, output_dir, feature, 
                         zip_output_path=None, required_left_mm=0.84, required_top_mm=0.0):
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    gray_values = {
        "ALL": {'Root': 25000, 'Nerve': 25000, 'Enamel': 25000},
        "Nerve": {'Root': 500, 'Nerve': 5000, 'Enamel': 1500},
        "Enamel": {'Root': 500, 'Nerve': 1500, 'Enamel': 25000},
        "Root": {'Root': 25000, 'Nerve': 1500, 'Enamel': 500}
    }.get(feature, {})

    if not gray_values:
        print(f"Unknown feature: {feature}")
        return []

    for filename in sorted(os.listdir(dicom_dir)):
        if not filename.endswith(".dcm"):
            continue

        dicom_path = os.path.join(dicom_dir, filename)
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array.copy().astype(np.int32)
        h, w = arr.shape

        # Enhanced metadata handling
        try:
            pixel_spacing = ds.PixelSpacing
        except AttributeError:
            # Fallback based on image data: 20.84mm = 104px → 0.2004 mm/px
            pixel_spacing = [0.2004, 0.2004]

        # Z-axis compensation (Slice Location)
        try:
            slice_position = float(ds.SliceLocation)
        except:
            slice_position = 0.0

        # Dynamic padding calculation
        pad_left_px = int(round(required_left_mm / pixel_spacing[1]))
        pad_top_px = int(round(required_top_mm / pixel_spacing[0]))

        # Apply Z-axis compensation
        if abs(slice_position) > 0.1:
            z_compensation = int(round(slice_position / pixel_spacing[0]))
            pad_top_px += z_compensation

        # File matching logic
        base = os.path.splitext(filename)[0]
        if (match := re.search(r'\d+', base)) is None:
            continue

        mask_key = f"mask_{match.group()}.png"
        if mask_key not in annotation_dict:
            continue

        # Apply annotations
        for label, coords in annotation_dict[mask_key].items():
            if (shade := gray_values.get(label)) is None:
                continue
            
            for x, y in coords:
                adj_x = x + pad_left_px
                adj_y = y + pad_top_px
                
                if 0 <= adj_x < w and 0 <= adj_y < h:
                    arr[adj_y, adj_x] = shade
                else:
                    print(f"Coordinate out of bounds: ({adj_x}, {adj_y})")

        # Save modified DICOM
        if np.issubdtype(ds.pixel_array.dtype, np.integer):
            arr = np.clip(arr, np.iinfo(ds.pixel_array.dtype).min, 
                         np.iinfo(ds.pixel_array.dtype).max)
        ds.PixelData = arr.astype(ds.pixel_array.dtype).tobytes()

        out_path = os.path.join(output_dir, f"{base}_annot.dcm")
        ds.save_as(out_path)
        saved_files.append(out_path)

    if zip_output_path:
        with ZipFile(zip_output_path, 'w') as zipf:
            for f in saved_files:
                zipf.write(f, os.path.basename(f))

    return saved_files

# ------------ Streamlit UI ----------------
st.title("🦷 Dental DICOM Annotator - Case 1")
tab1, = st.tabs(["📁 Case 1: Full DICOM Series"])

with tab1:
    selected_structure = st.selectbox("Choose annotation structure:", 
                                    ["Nerve", "Enamel", "Root", "ALL"])
    uploaded_files = st.file_uploader("Upload DICOM series", 
                                    type=["dcm"], accept_multiple_files=True)

    if uploaded_files and not st.session_state.processing_done:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Directory setup
            dicom_dir = os.path.join(tmpdir, "dicoms")
            mask_dir = os.path.join(tmpdir, "masks")
            os.makedirs(dicom_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            # Save uploaded files
            for file in uploaded_files:
                with open(os.path.join(dicom_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())

            # Process DICOM series
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)
            
            # Generate masks
            for i, filename in enumerate(sorted(os.listdir(dicom_dir))):
                dicom_path = os.path.join(dicom_dir, filename)
                idx = re.search(r'\d+', filename).group()
                
                # Process DICOM
                img_arr, _ = dicom_to_png(dicom_path)
                mask_path = os.path.join(mask_dir, f"mask_{idx}.png")
                predict_and_save_mask(model, img_arr, mask_path)
                
                progress_bar.progress((i+1)/total_files)

            # Build annotations
            annotation_dict = build_annotation_dictionary(mask_dir)
            
            # Annotate DICOMs
            zip_path = os.path.join(tmpdir, "annotated_dicoms.zip")
            output_folder = os.path.join(tmpdir, "annotated")
            
            annotate_dicom_series(
                annotation_dict,
                dicom_dir,
                output_folder,
                selected_structure,
                zip_output_path=zip_path,
                required_left_mm=1.84,  # From image data
                required_top_mm=2.0
            )

            # Store results
            with open(zip_path, "rb") as f:
                st.session_state.annotated_zip = f.read()
            
            st.session_state.processing_done = True
            progress_bar.empty()

    if st.session_state.processing_done:
        st.download_button(
            "⬇ Download Annotated Series",
            data=st.session_state.annotated_zip,
            file_name="annotated_series.zip",
            mime="application/zip"
        )
        if st.button("Reset Processor"):
            st.session_state.processing_done = False
            st.experimental_rerun()
