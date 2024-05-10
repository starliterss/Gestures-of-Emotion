import os
import tensorflow as tf
import numpy as np
from keras.layers import Layer
import random
from keras.models import Model
from keras.layers import Layer,Conv2D, Dense, MaxPooling2D, Input, Flatten
import cv2
from keras.models import model_from_json

## Face Rec ##

class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)



def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


def verify(model, detection_threshold, verification_threshold):
    results = []
    

    for image in os.listdir('FacePics/imagemyvalid'):
        input_img = preprocess('FacePics/INput/input_image.jpg')
        validation_img = preprocess(os.path.join('FacePics/imagemyvalid/', image))
        # print(validation_img)

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
        
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / 104 
    verified = verification >  verification_threshold
    return results,verified

def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

def make_siamese_model():
    
    embedding = make_embedding()

    input_image = Input(name = "input_img", shape=(100,100,3))
    validation_image = Input(name = "validation_img", shape=(100,100,3))
    
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image)) 
    
    classifier = Dense(1,activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs = classifier, name='SiameseNetwork') 


## Game Control ##

MIN_NUMBER = 1
MAX_NUMBER = 5
user_score = 0
computer_score = 0  # Initialize computer_score

def generate_computer_number():
    return random.randint(MIN_NUMBER, MAX_NUMBER)

def play_inning(user_input):
    global user_score, computer_score  # Include computer_score in global scope

    computer_number = generate_computer_number()

    print("Computer's number:", computer_number)

    if user_input < MIN_NUMBER or user_input > MAX_NUMBER:
        print("Invalid input. Choose a number between 1 and 5.")
        return

    if user_input == computer_number:
        print("You're out!")
    else:
        user_score += user_input
        print("Your current score:", user_score)

## Emotion Detector ##

json_file = open("emotiondetectornew1.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetectornew1.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
def EmotionDetector(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    for (p,q,r,s) in faces:
        image = gray[q:q+s,p:p+r]
        cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
        image = cv2.resize(image,(48,48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        # print("Predicted Output:", prediction_label)
        # cv2.putText(im,prediction_label)
        # cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        return prediction_label
 

