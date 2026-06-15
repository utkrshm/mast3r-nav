"""
File that processes the GoStanford 2 images from fisheye perspective to equirectangular perspective

Before using, set the locations for input and output directories
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from natsort import natsorted
import os


# Ricoh Theta S uses 2 190-deg FOV fisheye cameras (10 degree is for overlap)
def get_transformation_maps(h, w, hfov: float = 180):
    out_h = h
    out_w = 2*w
    
    cx = w / 2
    cy = h /2
    R = min(h, w) / 2
    
    xs = np.arange(out_w, dtype=np.float32)
    ys = np.arange(out_h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)
    
    # lon = (x_grid / out_w) * 2 * np.pi - np.pi
    # lat = (y_grid / out_h) * np.pi - (np.pi / 2) 
    
    # theta = (np.pi / 2) - lat
    # r = (theta / (np.radians(hfov) / 2)) * R
    # phi = lon - (np.pi / 2)

    lon = (x_grid / out_w) * 2.0 * np.pi - np.pi
    lat = (np.pi / 2.0) - (y_grid / out_h) * np.pi 
    
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    
    z = np.cos(lat) * np.cos(lon)    
    z = np.clip(z, -1.0, 1.0)
    
    theta = np.arccos(z)
    phi = np.arctan2(-y, x) 
    r = (theta / (np.radians(hfov) / 2.0)) * R
    
    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy + r * np.sin(phi)).astype(np.float32)
    
    return map_x, map_y

def process_imgs(img_list: List[Path], out_dir: Path, fov=180):
    """This function assumes that the dimensions for all the images will be the same throughout the batch"""
    h, w = cv2.imread(img_list[0]).shape[:2]
    
    map_x, map_y = get_transformation_maps(h, w, hfov=fov)
    for img_path in img_list:
        img_name= img_path.name
        img = cv2.imread(img_path)
        t_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Crop the center, to avoid capturing the black regions of the sides
        t_img = t_img[:, 63:(255-64), :]
        cv2.imwrite(out_dir / img_name, t_img)

if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).parent.parent / "data" / "gs2" / "img_L_1preF"
    assert BASE_DIR.exists(), KeyError("Input directory does not exist, please verify the path")

    start_file = "img_1pre_1_20_LF.jpg"
    end_file = "img_1pre_1_193_LF.jpg"
    
    skip_files = []     # Names of files to skip
    
    files = natsorted(list(BASE_DIR.iterdir()))
    start_idx = files.index(BASE_DIR / start_file)
    end_idx = files.index(BASE_DIR / end_file)
    
    assert start_idx != -1 and end_idx != -1, KeyError("start_file or end_file not in input directory, please verify")

    # The "images" folder is there to maintain consistency with how data from HM3D is sent to the mapper
    BASE_OUT_DIR = Path(__file__).parent.parent / "data" / "gs2_modified"
    OUT_DIR = BASE_OUT_DIR / BASE_DIR.name / "images"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # print(OUT_DIR, OUT_DIR.exists())
    img_files = [img_path for img_path in files[start_idx:end_idx+1] if img_path.name not in skip_files]
    
    print(BASE_DIR, OUT_DIR)
    process_imgs(img_files, OUT_DIR, fov=170)
