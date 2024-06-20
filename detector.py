 

# detector.py

from pathlib import Path
import face_recognition
import pickle
from collections import Counter
import pdb
from PIL import Image, ImageDraw

# sets the default path for encodings to be in our output folder
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


# Iterates through training directory and encodes the faces we have asked it to find
#only need to call it if we change the training directory
def encode_known_faces( model: str = 'hog', encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model = model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
             names.append(name)
             encodings.append(encoding)
# saving the encodings to the disk for later access
    name_encodings = {"names" : names, "encodings" : encodings}
    with encodings_location.open(mode = "wb") as f:
        pickle.dump(name_encodings, f)

#function to display face in image
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

#Next step is to recognize faces in unseen images
#open and load saved face encoding and load image which you want to recognize
#compare new encoded image with existing encoding database
#return name of match


def recognize_faces(image_location: str, model: str = 'hog', encodings_location : Path = DEFAULT_ENCODINGS_PATH) -> None:

    #load encodings
    with encodings_location.open(mode = "rb") as f:
        loaded_encodings = pickle.load(f)

    #load target face
    input_image = face_recognition.load_image_file(image_location)
    

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    #draw box around face
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    #compare input encodings with the original encodings.
    #print name from dictionary or default 'unknown'
    for bounding_box, unkown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unkown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        #print(name, bounding_box)
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


#function to compare the two encodings 
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)

    votes = Counter(
        name 
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

#encode_known_faces
recognize_faces('unknown.jpeg')



    
