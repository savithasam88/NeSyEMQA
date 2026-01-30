''' vstream env: Download model flash-vstream-qwen
from huggingface_hub import snapshot_download
local_path = "/users/sbsh670/archive/models"
repo_id = "zhang9302002/Flash-VStream-Qwen-7b"
snapshot_download(repo_id=repo_id, local_dir=local_path)
'''

#Unzip all video files
import zipfile
import tarfile
import os


def tar_files_in_folder(root_folder):
    # Walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file_name in filenames:
            if file_name.endswith(".tar.gz"):
                tar_path = os.path.join(dirpath, file_name)
                extract_path = os.path.join(dirpath, file_name[:-7])  # Remove '.tar.gz' for folder name

                os.makedirs(extract_path, exist_ok=True)

                with tarfile.open(tar_path, "r:gz") as tar_ref:
                    tar_ref.extractall(extract_path)

                print(f"Extracted {tar_path} to {extract_path}")


def unzip_files_in_folder(root_folder):
    # Walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file_name in filenames:
            if file_name.endswith(".zip"):
                zip_path = os.path.join(dirpath, file_name)
                extract_path = os.path.join(dirpath, file_name[:-4])  # Create folder with zip name

                # Create folder if it doesn't exist
                os.makedirs(extract_path, exist_ok=True)

                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
        
                print(f"Extracted {zip_path} to {extract_path}")

path_MVBench = "/users/sbsh670/archive/data/eval_video/MVBench/video"
path_LLava_video = "/users/sbsh670/archive/data/train/LLaVA-Video-178K"

unzip_files_in_folder(path_MVBench)
#tar_files_in_folder(path_LLava_video)



''' TEST Qwen
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
 
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/users/sbsh670/archive/openeqaDataFrames/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE/00024-rgb.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
#inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
'''
