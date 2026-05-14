from glob import glob
from tqdm import tqdm
import os
from os.path import join, isfile, basename
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime
from PIL import Image
import SimpleITK as sitk # THE FIX: Using SimpleITK for Native Medical Geometries
import torch
import torch.multiprocessing as mp
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2
import yaml

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def resize_rgb(array, image_size):
    d, h, w = array.shape[:3]
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_rgb = Image.fromarray(array[i].astype(np.uint8))
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def grayscale2rgb(array):
    if len(array.shape) > 3:
        return array

    rgb_array = np.zeros((*array.shape, 3))
    for i in range(array.shape[0]):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        rgb_array[i] = np.array(img_rgb)

    return rgb_array


@torch.inference_mode()
def infer_3d(predictor, img_path, parsed_data, propagate, model_cfg, output_path):
    print(f'Inferring {img_path}')
    
    # THE FIX: Read NIfTI directly to preserve spatial alignment
    img_sitk = sitk.ReadImage(img_path)
    img_3D = sitk.GetArrayFromImage(img_sitk)  # Natively loads as (D, H, W)
    
    if np.max(img_3D) >= 256:
        img_3D = (img_3D - np.min(img_3D)) / (np.max(img_3D) - np.min(img_3D)) * 255
        img_3D = img_3D.astype(np.int16)
        
    img_3D = grayscale2rgb(img_3D)
    D, H, W = img_3D.shape[:3]
    segs_3D = np.zeros(img_3D.shape[:3], dtype=np.uint8)
    
    # Extract data sent via JSON from the FastAPI server
    boxes_3D = np.array(parsed_data.get('boxes', []))
    z_range = parsed_data.get('z_range', [0, D-1, int(D/2)])
    z_indices, slice_idx = np.array(z_range[:2]), z_range[2]
    
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]
    
    with open(join('sam2', model_cfg), 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        image_size = yaml_data['model']['image_size']
        
    img_resized = resize_rgb(img_3D, image_size)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    
    z_mids = []

    # MIDDLE SLICE ONLY MODE
    if not propagate:
        img = img_3D[slice_idx] / 255.
        img = img.astype(np.float32)
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes_3D, multimask_output=False,)
        if len(masks.shape) == 3: # single bounding box
            masks = [masks]
        for idx, mask in enumerate(masks, start=1):
            segs_3D[slice_idx, (mask[0] > 0.0)] = idx
            
        # THE FIX: Wrap result back into ITK to apply the spatial metadata
        out_sitk = sitk.GetImageFromArray(segs_3D)
        out_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(out_sitk, output_path)
        print('Middle Slice Mask Calculated and saved to NIfTI')
        return None

    # PROPAGATE (VIDEO) MODE
    for idx, points in enumerate(boxes_3D, start=1):
        # We assume GTs aren't strictly necessary for a pure bounding-box run unless doing exact refinement
        # Assuming z_mid is calculated from the bounds
        z_min = z_indices.min() if z_indices.size > 0 else 0
        z_max = z_indices.max() if z_indices.size > 0 else D - 1
        
        img = img_resized[z_min:(z_max+1)]
        z_mid = int(img.shape[0]/2)
        z_mids.append(z_mid)
        
        ann_frame_idx = slice_idx - z_min

        print('analyzed image size', img.shape, 'mid idx', z_mid)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(img, video_height, video_width)
            
            # Using Add New Box instead of New Mask to avoid needing a GT mask
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                inference_state=inference_state, 
                frame_idx=ann_frame_idx, 
                obj_id=1, 
                box=points
            )
            segs_3D[slice_idx, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                print(f"Forward pass: {out_frame_idx}")
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
                
            predictor.reset_state(inference_state)
            
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                inference_state=inference_state, 
                frame_idx=ann_frame_idx, 
                obj_id=1, 
                box=points
            )
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                print(f"Reverse pass: {out_frame_idx}")
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
                
            predictor.reset_state(inference_state)

    print(f"Unique segments found: {np.unique(segs_3D)}")
    
    # THE FIX: Wrap result back into ITK to apply the spatial metadata
    out_sitk = sitk.GetImageFromArray(segs_3D)
    out_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(out_sitk, output_path)

    return inference_state


@torch.inference_mode()
def improve_3d(predictor, inference_state, img_path, parsed_data, output_path):
    img_sitk = sitk.ReadImage(img_path)
    img_3D = sitk.GetArrayFromImage(img_sitk)
    segs_3D = np.zeros(img_3D.shape[:3], dtype=np.uint8)
    
    points_addition = np.array(parsed_data.get('points_addition', []))
    points_subtraction = np.array(parsed_data.get('points_subtraction', []))
    
    labels = np.array([1]*points_addition.shape[0] + [0]*points_subtraction.shape[0], dtype=np.int32)
    
    if points_addition.shape[0] == 0:
        points = points_subtraction
    elif points_subtraction.shape[0] == 0:
        points = points_addition
    else:
        points = np.vstack((points_addition, points_subtraction))
        
    box = np.array(parsed_data['bboxes'][0], np.float32)
    z_min = min(parsed_data['zrange'])
    z_mid_orig = int(points[0, -1])
    ann_frame_idx = z_mid_orig - z_min

    points = points[:,:-1].astype(np.float32) # dropping 3rd dimension

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
        )
        segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = 1
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
            
        predictor.reset_state(inference_state)
        
        _, _, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=box,
        )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
            
        predictor.reset_state(inference_state)

    # THE FIX: Wrap result back into ITK to apply the spatial metadata
    out_sitk = sitk.GetImageFromArray(segs_3D)
    out_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(out_sitk, output_path)

    return inference_state


def perform_inference(checkpoint, cfg, img_path, parsed_data, output_path, propagate=True):
    predictor = build_sam2_video_predictor_npz(cfg, checkpoint) if propagate else SAM2ImagePredictor(build_sam2(cfg, checkpoint, device="cuda"))
    inference_state = infer_3d(predictor, img_path, parsed_data, propagate, cfg, output_path)
    return predictor, inference_state


def improve_inference(img_path, parsed_data, output_path, predictor_state):
    predictor, inference_state = predictor_state['predictor'], predictor_state['inference_state']
    inference_state = improve_3d(predictor, inference_state, img_path, parsed_data, output_path)
    return predictor, inference_state


if __name__ == '__main__':
    # ---------------------------------------------------------
    # TEST FUNCTIONALITY
    # ---------------------------------------------------------
    print('Running installation test...')
    
    # 1. Create a dummy NIfTI file since we moved away from .npz
    test_img_path = 'test_dummy_vol.nii.gz'
    test_out_path = 'test_dummy_mask.nii.gz'
    
    if not os.path.isfile(test_img_path):
        print(f"Creating dummy volume {test_img_path} for testing...")
        # Create a 10-slice blank volume with a small white square as a "lesion"
        dummy_vol = np.zeros((10, 128, 128), dtype=np.uint8)
        dummy_vol[5, 50:80, 50:80] = 255 
        dummy_sitk = sitk.GetImageFromArray(dummy_vol)
        sitk.WriteImage(dummy_sitk, test_img_path)
        
    # 2. Mock the parsed JSON data that the FastAPI server would normally send
    mock_parsed_data = {
        'boxes': [[[50, 50, 80, 80]]],
        'z_range': [0, 9, 5]  # z_min, z_max, slice_idx
    }
    
    # 3. Run the inference
    # propagate=False tests just the middle slice (same as your original test script)
    try:
        perform_inference(
            checkpoint='checkpoints/MedSAM2_latest.pt', 
            cfg='MedSAM2_tiny512.yaml', 
            img_path=test_img_path, 
            parsed_data=mock_parsed_data, 
            output_path=test_out_path,
            propagate=False
        )
        print('Server is installed and NIfTI pipeline works perfectly!')
    except Exception as e:
        print(f"Server installation test failed: {e}")