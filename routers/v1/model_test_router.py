from fastapi import APIRouter, UploadFile, File, status, HTTPException
from PIL import Image
import io
import os
import uuid
from fastapi.responses import JSONResponse
# functions calling
from utils.model_load import load_keras_model
from utils.image_preprocessing import preprocess_image
# load the model-->function calling
# function-->utils-->load_keras_model
model = load_keras_model("model.h5")

output_dir = "Files"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

router = APIRouter(
    prefix="/predict",
    tags=["Image Classification"]
)


@router.post("/")
async def get_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Image file required (jpg, png, jpeg)")

    # Read the file
    file_data = await file.read()

    try:

        # Open the image
        image_uploaded = Image.open(io.BytesIO(file_data))
        # Generate a unique file name with extension
        file_extension = file.filename.split('.')[-1]

        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        file_location = os.path.join(output_dir, unique_filename)

        # Save the image
        image_uploaded.save(file_location)

        # Preprocess the image function-->utils->process_image
        img_array = preprocess_image(image_uploaded)

        # Make a prediction
        prediction = model.predict(img_array)

        if prediction[0] > 0.5:
            detail = "There is a Dog image"
        else:
            detail = "There is a Cat image"

        return JSONResponse(content={"filename": file.filename, "message": "Prediction successful!", "detail": detail})

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unable to process the image: {str(e)}")
