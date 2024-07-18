import face_recognition as fr
import cv2
import numpy as np
import os

# Path to the training images
path = "./train/"

known_names = []
known_name_encodings = []

# Load training images and their encodings
images = os.listdir(path)
for _ in images:
    image_path = os.path.join(path, _)
    image = fr.load_image_file(image_path)
    encodings = fr.face_encodings(image)
    
    # Check if there are any face encodings found
    if encodings:
        encoding = encodings[0]
        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())
    else:
        print(f"No faces found in the image {image_path}")

print(known_names)

# Path to the test image
test_image_path = "./test/test.jpg"
image = cv2.imread(test_image_path)
# Convert the image color to RGB as face_recognition uses RGB format
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find face locations and encodings in the test image
face_locations = fr.face_locations(rgb_image)
face_encodings = fr.face_encodings(rgb_image, face_locations)

# Iterate through each face found in the test image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = "Unknown"

    # Use the known face with the smallest distance to the new face
    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_names[best_match_index]

    # Draw a rectangle around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    # Draw a label with a name below the face
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display the resulting image
cv2.imshow("Result", image)
cv2.imwrite("./output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
