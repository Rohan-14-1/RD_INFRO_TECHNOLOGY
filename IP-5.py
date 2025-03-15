import cv2
import numpy as np
import os
import socket
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.datasets import fetch_lfw_people

def check_internet():
    """Check if the system has an active internet connection."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False

# Load LFW dataset from sklearn
def load_lfw_dataset():
    if not check_internet():
        print("No internet connection! Ensure dataset is manually placed.")
        return None
    print("Downloading LFW dataset...")
    lfw = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=1)
    return lfw

lfw_dataset = load_lfw_dataset()
if lfw_dataset is None:
    print("Dataset download failed. Please check your connection.")
    exit()

def get_first_image():
    """Retrieve the first available image from LFW dataset."""
    if lfw_dataset.images.size > 0:
        return lfw_dataset.images[0].astype(np.uint8)
    return None

# Load Face Detection Model (MTCNN)
mtcnn = MTCNN(keep_all=True)

# Load Face Recognition Model (FaceNet)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def detect_faces(image):
    """Detect faces in an image using MTCNN."""
    boxes, _ = mtcnn.detect(image)
    return boxes

def extract_face_embedding(image):
    """Extract facial embeddings using FaceNet."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    embedding = resnet(image)
    return embedding.detach().numpy()

def main():
    sample_image = get_first_image()
    if sample_image is None:
        print("No images found in the dataset!")
        return
    
    print("Using first image from LFW dataset.")
    boxes = detect_faces(sample_image)
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = sample_image[y1:y2, x1:x2]
            embedding = extract_face_embedding(face)
            print("Face Embedding:", embedding)
            cv2.rectangle(sample_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Face Detection", cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()