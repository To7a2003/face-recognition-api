import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/verify/")
async def verify_all_images(
    ID_image: UploadFile = File(..., alias="ID_image"),
    reference_image: UploadFile = File(..., alias="reference_image"),
    test_image: UploadFile = File(..., alias="test_image")
):
    def preprocess(image_bytes):
        img_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        img = np.expand_dims(img, axis=0) / 255.0
        return img

    # قراءة الصور ومعالجتها
    id_bytes = await ID_image.read()
    ref_bytes = await reference_image.read()
    test_bytes = await test_image.read()

    id_img = preprocess(id_bytes)
    ref_img = preprocess(ref_bytes)
    test_img = preprocess(test_bytes)

    if id_img is None or ref_img is None or test_img is None:
        return JSONResponse(content={"message": "One or more images could not be loaded."}, status_code=400)

    # تحميل الموديل داخل الفانكشن فقط وقت الطلب
    model = tf.keras.models.load_model("Final_face_recognition_cnn_model.h5")

    # استخراج التمثيل (encoding)
    id_encoding = model.predict(id_img, verbose=0)
    reference_encoding = model.predict(ref_img, verbose=0)
    test_encoding = model.predict(test_img, verbose=0)

    # التحقق
    threshold = 0.7
    is_verified = any(
        np.linalg.norm(enc - test_encoding) < threshold
        for enc in [id_encoding, reference_encoding]
    )

    return {"is_verified": is_verified}
