"""
Pipeline that processes fisheye images into a standard rectilinear (pinhole) center crop,
and then immediately stitches them into a video using FFmpeg with a 0.5s frame duration.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List
from natsort import natsorted
import os
import subprocess

def get_rectilinear_maps(in_h: int, in_w: int, out_size: int, fish_fov: float = 180.0, rect_fov: float = 90.0):
    """
    Generates mapping coordinates to extract a rectilinear crop from a fisheye image.
    """
    cx_fish = in_w / 2.0
    cy_fish = in_h / 2.0
    
    r_fish_max = min(in_h, in_w) / 2.0 
    
    out_w = out_size
    out_h = out_size
    cx_rect = out_w / 2.0
    cy_rect = out_h / 2.0

    f_rect = cx_rect / np.tan(np.radians(rect_fov) / 2.0)

    xs = np.arange(out_w, dtype=np.float32)
    ys = np.arange(out_h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)

    x_rect = x_grid - cx_rect
    y_rect = y_grid - cy_rect

    rho = np.sqrt(x_rect**2 + y_rect**2)
    theta = np.arctan2(rho, f_rect)
    phi = np.arctan2(y_rect, x_rect)

    theta_max = np.radians(fish_fov) / 2.0
    r_fish = (theta / theta_max) * r_fish_max

    map_x = (cx_fish + r_fish * np.cos(phi)).astype(np.float32)
    map_y = (cy_fish + r_fish * np.sin(phi)).astype(np.float32)

    return map_x, map_y

def process_imgs(img_list: List[Path], out_dir: Path, fish_fov: float = 180.0, rect_fov: float = 90.0, out_resolution: int = 512):
    """Processes a batch of images using the calculated maps."""
    if not img_list:
        print("No images found to process.")
        return

    h, w = cv2.imread(str(img_list[0])).shape[:2]
    
    print(f"Generating transformation maps for {len(img_list)} images...")
    map_x, map_y = get_rectilinear_maps(h, w, out_size=out_resolution, fish_fov=fish_fov, rect_fov=rect_fov)
    
    for img_path in img_list:
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        
        t_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        cv2.imwrite(str(out_dir / img_name), t_img)
    
    print("Cropping complete.")

def create_video(image_dir: Path, output_video_path: Path, frame_duration: float = 0.5):
    """Generates a video from the processed frames using FFmpeg concat demuxer."""
    print(f"Stitching video from frames in {image_dir}...")
    
    files = natsorted([f.name for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png']])
    
    if not files:
        print("No images found to create video.")
        return

    # 1. Generate the frames.txt list inside the output directory
    frames_txt_path = image_dir / "frames.txt"
    with open(frames_txt_path, "w") as f:
        for file in files:
            f.write(f"file '{file}'\n")
            f.write(f"duration {frame_duration}\n")
        
        # FFmpeg requirement: declare the last file again without a duration
        f.write(f"file '{files[-1]}'\n")

    # 2. Execute the FFmpeg command
    fps = 1.0 / frame_duration
    
    cmd = [
        "ffmpeg", 
        "-y",               # Overwrite output file if it already exists
        "-f", "concat", 
        "-safe", "0", 
        "-i", "frames.txt", 
        "-c:v", "libx264", 
        "-vf", f"fps={fps}", 
        "-pix_fmt", "yuv420p", 
        str(output_video_path.resolve())
    ]
    
    try:
        # Run the command with cwd set to image_dir so frames.txt finds the images relative to itself
        subprocess.run(cmd, cwd=image_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Video successfully saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg. Check your installation. Details: {e}")
    except FileNotFoundError:
        print("FFmpeg command not found. Please ensure FFmpeg is installed and added to your system PATH.")


if __name__ == "__main__":
    
    # 1. Define Paths
    BASE_DIR = Path(__file__).parent.parent / "data" / "gs2" / "img_L_1preF" 
    assert BASE_DIR.exists(), KeyError("Input directory does not exist, please verify the path")

    start_file = "img_1pre_1_20_LF.jpg"
    end_file = "img_1pre_1_193_LF.jpg"
    skip_files = [] 
    
    files = natsorted(list(BASE_DIR.iterdir()))
    start_idx = files.index(BASE_DIR / start_file)
    end_idx = files.index(BASE_DIR / end_file)
    assert start_idx != -1 and end_idx != -1, KeyError("start_file or end_file not in input directory, please verify")

    BASE_OUT_DIR = Path(__file__).parent.parent / "data" / "gs2_modified"
    OUT_DIR = BASE_OUT_DIR / BASE_DIR.name / "images"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    img_files = [img_path for img_path in files[start_idx:end_idx+1] if img_path.name not in skip_files]
    
    print(f"Input: {BASE_DIR}")
    print(f"Output frames: {OUT_DIR}")
    
    # 2. Process images to a 512x512 rectilinear crop
    process_imgs(img_files, OUT_DIR, fish_fov=180.0, rect_fov=170.0, out_resolution=512)
    
    # 3. Compile the processed frames into a video
    video_output_file = BASE_OUT_DIR / BASE_DIR.name / "sequence_output.mp4"
    create_video(image_dir=OUT_DIR, output_video_path=video_output_file, frame_duration=0.5)