from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Allow up to 50 MB

def preprocess_image(image):
    fixed_width = 800
    height, width = image.shape[:2]
    scale = fixed_width / width
    resized_image = cv2.resize(image, (fixed_width, int(height * scale)))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return resized_image, gray_image

def match_keypoints(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches, keypoints1, keypoints2

def stitch_images_homography(img1, img2):
    good_matches, kp1, kp2 = match_keypoints(img1, img2)
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = img1.shape[:2]
        stitched_img = cv2.warpPerspective(img2, H, (width * 2, height))
        stitched_img[0:height, 0:width] = img1
        return stitched_img
    else:
        raise ValueError("Not enough matches found for stitching!")

def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return image[y:y+h, x:x+w]

@app.route('/stitch', methods=['POST'])
def stitch_images():
    data = request.json
    images_data = data.get('images', [])
    if len(images_data) < 2:
        return jsonify({"error": "At least two images are required for stitching"}), 400
    images = []
    for img_data in images_data:
        try:
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed_img, _ = preprocess_image(img_cv)
            images.append(processed_img)
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 400
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched = stitcher.stitch(images)
        if status != cv2.Stitcher_OK:
            stitched = images[0]
            for i in range(1, len(images)):
                stitched = stitch_images_homography(stitched, images[i])
        stitched = crop_black_borders(stitched)
        _, buffer = cv2.imencode('.jpg', stitched)
        stitched_image_bytes = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"stitched_image": stitched_image_bytes})
    except Exception as e:
        return jsonify({"error": f"Image stitching failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
