from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import filetype
import infer

app = FastAPI()

# # Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor
predictor = infer.initialize()

# Create the router
router = APIRouter()

# Define the class for the endpoint
class VideoUploader:
    @staticmethod
    async def upload_video(video: UploadFile = File(...)):
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                contents = await video.read()
                temp_file.write(contents)
                temp_file_path = temp_file.name

            # Check if the uploaded file is a video
            kind = filetype.guess(temp_file_path)
            if kind is None or not kind.mime.startswith('video/'):
                raise HTTPException(status_code=400, detail="The file is not a video")

            prediction = await predictor(temp_file_path)

            # Delete the temporary file
            shutil.rmtree(tempfile.gettempdir(), ignore_errors=True)

            return JSONResponse(content=prediction, status_code=200)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Add the endpoint to the router
router.add_api_route("/upload_video", VideoUploader.upload_video, methods=["POST"])

# Include the router in the main app
app.include_router(router)