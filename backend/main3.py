from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import face_recognition
import cv2
import numpy as np
import os
import uuid
import pymongo
from bson import Binary
import pickle
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, field_validator, Field
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Face Recognition API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PersonDetails(BaseModel):
    first_name: str
    last_name: str
    age: Optional[str] = Field(None)
    height: Optional[str] = Field(None)
    weight: Optional[str] = Field(None)

    @field_validator('first_name', 'last_name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

class FaceRecognitionSystem:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="face_recognition"):
        try:
            self.client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.server_info()
            self.db = self.client[db_name]
            self.people_collection = self.db['people']
            self.faces_collection = self.db['faces']
            
            # Create indexes
            self.people_collection.create_index("person_id", unique=True)
            self.faces_collection.create_index("person_id", unique=True)
            
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_ids = []
            self.load_existing_data()
            logger.info("Face Recognition System initialized successfully")
        except Exception as e:
            logger.error(f"MongoDB Connection Error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database connection failed")

    def load_existing_data(self):
        try:
            cursor = self.faces_collection.find({})
            for doc in cursor:
                face_encoding = pickle.loads(doc['encoding'])
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(doc['name'])
                self.known_face_ids.append(doc['person_id'])
            logger.info(f"Loaded {len(self.known_face_encodings)} existing face encodings")
        except Exception as e:
            logger.error(f"Error loading existing data: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load existing data")

    async def validate_image(self, image_data: bytes) -> bool:
        try:
            img = Image.open(io.BytesIO(image_data))
            
            # Check file size (max 5MB)
            if len(image_data) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image too large (max 5MB)")
            
            # Check minimum dimensions
            if img.size[0] < 200 or img.size[1] < 200:
                raise HTTPException(status_code=400, detail="Image too small (min 200x200)")
                
            return True
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    async def process_image(self, image_data: bytes):
        try:
            # Validate image
            await self.validate_image(image_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            if len(face_locations) > 1:
                raise HTTPException(status_code=400, detail="Multiple faces detected in image")
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=2)
            
            if not face_encodings:
                raise HTTPException(status_code=400, detail="Could not encode face features")
                
            return face_encodings[0]
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image")

    async def recognize_or_add(self, image: UploadFile):
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        try:
            contents = await image.read()
            face_encoding = await self.process_image(contents)
            
            if not self.known_face_encodings:
                return {"status": "not_found"}

            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=0.5
            )
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    person_id = self.known_face_ids[best_match_index]
                    person_details = self.people_collection.find_one(
                        {'person_id': person_id},
                        {'_id': 0}
                    )
                    
                    if person_details:
                        confidence = 1 - face_distances[best_match_index]
                        person_details['confidence'] = f"{confidence:.2%}"
                        return {
                            "status": "recognized",
                            "person": person_details
                        }

            return {"status": "not_found"}
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error during recognition: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

    async def find_existing_person(self, face_encoding):
        """Find if a face encoding matches anyone in the database"""
        if not self.known_face_encodings:
            return None, None
            
        matches = face_recognition.compare_faces(
            self.known_face_encodings, 
            face_encoding,
            tolerance=0.5
        )
        
        if True in matches:
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                person_id = self.known_face_ids[best_match_index]
                person_details = self.people_collection.find_one(
                    {'person_id': person_id},
                    {'_id': 0}
                )
                
                if person_details:
                    confidence = 1 - face_distances[best_match_index]
                    return person_details, f"{confidence:.2%}"
                    
        return None, None

    async def add_person(self, image: UploadFile, first_name: str, last_name: str, 
                         age: Optional[str] = None, height: Optional[str] = None, 
                         weight: Optional[str] = None):
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        try:
            # Validate names
            if not first_name.strip() or not last_name.strip():
                raise HTTPException(status_code=400, detail="First name and last name are required")
            
            contents = await image.read()
            face_encoding = await self.process_image(contents)
            
            # Check if face already exists and return existing person details if found
            existing_person, confidence = await self.find_existing_person(face_encoding)
            if existing_person:
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": "Face already registered in the system",
                        "existing_person": existing_person,
                        "confidence": confidence
                    }
                )
            
            person_id = str(uuid.uuid4())[:8]
            person_name = f"{first_name.strip()} {last_name.strip()}"

            # Store person details
            details_dict = {
                'person_id': person_id,
                'first_name': first_name.strip(),
                'last_name': last_name.strip(),
                'timestamp': datetime.now()
            }
            
            # Add optional fields if provided
            if age is not None and age.strip():
                details_dict['age'] = age.strip()
            if height is not None and height.strip():
                details_dict['height'] = height.strip()
            if weight is not None and weight.strip():
                details_dict['weight'] = weight.strip()
                
            self.people_collection.insert_one(details_dict)
            
            # Store face encoding
            self.faces_collection.insert_one({
                'person_id': person_id,
                'name': person_name,
                'encoding': Binary(pickle.dumps(face_encoding)),
                'timestamp': datetime.now()
            })
            
            # Update in-memory data
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(person_name)
            self.known_face_ids.append(person_id)
            
            logger.info(f"Added new person: {person_name} with ID: {person_id}")
            return {"status": "success", "person_id": person_id}
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error adding person: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to add person: {str(e)}")

# Initialize face recognition system
face_system = FaceRecognitionSystem()

@app.post("/recognize_or_add/")
async def recognize_or_add(image: UploadFile = File(...)):
    """
    Recognize a face in the uploaded image
    """
    return await face_system.recognize_or_add(image)

@app.post("/add_person/")
async def add_person(
    image: UploadFile = File(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    age: Optional[str] = Form(None),
    height: Optional[str] = Form(None),
    weight: Optional[str] = Form(None)
):
    """
    Add a new person to the system
    """
    try:
        return await face_system.add_person(
            image=image,
            first_name=first_name,
            last_name=last_name,
            age=age,
            height=height,
            weight=weight
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in add_person endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)