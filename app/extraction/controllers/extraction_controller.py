import base64
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.status import HTTP_400_BAD_REQUEST

from ..schemas.extraction_schema import ExtractionResponse
from ..services.extraction_service import extraction_service


router = APIRouter(prefix='/text_extraction', tags=['text_extraction'])

@router.post("/upload", response_model=ExtractionResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Process uploaded image and extract text
    """
    # Check file size (max 15 MB)
    if file.size > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail="File size exceeds 15 MB limit"
        )
    
    # Read file contents
    image_bytes = await file.read()
    
    # Preprocess image
    original_image, binary_image = await extraction_service.preprocess_image(image_bytes)
    
    # Extract text and regions
    text, text_regions = await extraction_service.extract_text_and_regions(binary_image)

    # Draw text regions on image
    result_image = await extraction_service.draw_text_regions(original_image, text_regions)
    
    # Convert result image to base64 for response
    image_base64 = await extraction_service.convert_image_to_base64_string(result_image)
    
    return ExtractionResponse(
        text=text.strip(),
        status="success",
        image=image_base64
    )