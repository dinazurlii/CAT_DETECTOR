import os
from PIL import Image
from pillow_heif import register_heif_opener

# Daftarkan HEIC supaya bisa dibuka via PIL
register_heif_opener()

input_dir = "cat_dataset"
output_dir = "processed_cats"

os.makedirs(output_dir, exist_ok=True)

TARGET = 224

valid_ext = (".jpg", ".jpeg", ".png", ".heic", ".HEIC", ".JPG", ".JPEG", ".PNG")

for cls in ["cute", "ugly"]:
    src = os.path.join(input_dir, cls)
    dst = os.path.join(output_dir, cls)
    os.makedirs(dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.endswith(valid_ext)]

    print(f"Processing {cls}: {len(files)} files")

    for f in files:
        try:
            img_path = os.path.join(src, f)

            # Open HEIC/JPG/PNG all with PIL automatically
            img = Image.open(img_path).convert("RGB")

            # Resize shortest side = 256
            w, h = img.size
            scale = 256 / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            # Center crop to 224x224
            w2, h2 = img.size
            left = (w2 - TARGET) // 2
            top = (h2 - TARGET) // 2
            img = img.crop((left, top, left + TARGET, top + TARGET))

            # pastikan save sebagai JPG
            save_name = os.path.splitext(f)[0] + ".jpg"
            img.save(os.path.join(dst, save_name), "JPEG")

        except Exception as e:
            print("Error:", f, e)

print("\nDONE! All photos processed + converted to JPG in processed_cats/")
