import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse



app = FastAPI()

@app.post("/verify_all_images")
async def verify_all_images(
    id_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...)
):
    try:
        model = tf.keras.models.load_model("Final_face_recognition_cnn_model.h5")

        async def process_image(uploaded_file):
            image_bytes = await uploaded_file.read()
            image_np = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image.")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))
            img = np.expand_dims(img, axis=0) / 255.0
            return model.predict(img, verbose=0)

        id_encoding = await process_image(id_image)
        reference_encoding = await process_image(reference_image)
        test_encoding = await process_image(test_image)

        threshold = 0.7
        is_verified = any(
            np.linalg.norm(enc - test_encoding) < threshold
            for enc in [id_encoding, reference_encoding]
        )

        return {"is_verified": bool(is_verified)}

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
