import os
from pathlib import Path
import tempfile
import base64
from PIL import Image
import io
import copy

temp_dir = Path(__file__).parent/"tmp"
temp_dir.mkdir(exist_ok=True)
# temp_dir = tempfile.mkdtemp()
def get_temp_file_with_prefix_suffix(temp_dir = temp_dir, prefix="", suffix=""):
    return tempfile.mktemp(suffix=suffix, prefix=prefix, dir=temp_dir)

# Convert image to base64
def image_to_base64_(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte)
        return img_base64.decode('utf-8')

def image_dict_to_base64(image_dict):
    image_dict = copy.deepcopy(image_dict)
    image_dict["img_base64"] = r"data:image/png;base64," + image_to_base64_(image_dict.pop("image_path"))
    image_dict["img_name"] = image_dict["image_name"]
    if "image_des" in image_dict:
        image_dict["des"] = image_dict["image_des"]
    if "image_res" in image_dict:
        image_dict["res"] = image_dict["image_res"]
    return image_dict


if __name__ == "__main__":
    temp_file = get_temp_file_with_prefix_suffix(suffix=".png")
    print(temp_file)