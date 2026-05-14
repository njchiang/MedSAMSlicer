from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
import shutil
import gc
import torch
from pathlib import Path
import json

from infer_MedSAM2_slicer import perform_inference, improve_inference

app = FastAPI(title="MedSAM2 Slicer Server")

# Global state to hold the heavy model
predictor_state = {
    'predictor': None,
    'inference_state': None
}

def clear_memory():
    """Forces PyTorch to release VRAM."""
    predictor_state['predictor'] = None
    predictor_state['inference_state'] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.post("/unload")
async def unload_model():
    """Endpoint to free up the GPU for your LLM."""
    clear_memory()
    return {"status": "Model unloaded, VRAM cleared."}

@app.post("/segment")
async def run_script(
    file: UploadFile = File(...),
    bboxes: str = Form("{}"), # Corrected to default to an empty JSON object
    checkpoint: str = Form(...),
    config: str = Form(...),
    propagate: bool = Form(True)
):
    """Expects a .nii.gz file and spatial prompts, returns a .nii.gz mask."""
    work_dir = Path("data/workspace")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = work_dir / file.filename
    output_path = work_dir / f"mask_{file.filename}"

    # Save the incoming NIfTI file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    parsed_bboxes = json.loads(bboxes)
    ckpt_path = f"checkpoints/{checkpoint}"

    predictor, inference_state = perform_inference(
        ckpt_path, config, str(input_path), parsed_bboxes, str(output_path), propagate=propagate
    )
    
    predictor_state['predictor'] = predictor
    predictor_state['inference_state'] = inference_state

    # FileResponse streams the output file directly back to the Slicer client
    return FileResponse(path=output_path, filename=f"mask_{file.filename}")

@app.post("/improve")
async def improve(
    file: UploadFile = File(...),
    points: str = Form("{}") # JSON string containing 'addition' and 'subtraction' points
):
    work_dir = Path("data/workspace")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = work_dir / file.filename
    output_path = work_dir / f"improved_mask_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    parsed_points = json.loads(points)

    predictor, inference_state = improve_inference(
        str(input_path), parsed_points, str(output_path), predictor_state
    )
    
    predictor_state['predictor'] = predictor
    predictor_state['inference_state'] = inference_state

    return FileResponse(path=output_path, filename=f"improved_mask_{file.filename}")

if __name__ == '__main__':
    # Run with uvicorn for high performance
    uvicorn.run(app, host='0.0.0.0', port=8080)
