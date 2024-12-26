import cv2
import mediapipe as mp
import numpy as np
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.models import load_model, Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout # type: ignore

import tensorflowjs as tfjs

class FingerPen:
    def __init__(self, camera_index, model_path, mapping_path, camera_flipped=True, hand_detect_thresh=0.7, hand_tracking_thresh=0.5):
        self.camera_flipped = camera_flipped
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=hand_detect_thresh,
            min_tracking_confidence=hand_tracking_thresh
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.camera = cv2.VideoCapture(camera_index)
        self.letter_points = [[]]
        self.last_point = None
        
        self.char_map = {}
        mapping = open(mapping_path, 'r')
        for i in mapping.readlines():
            c, v = i.replace('\n', '').split(' ')
            self.char_map[int(c)] = chr(int(v))
        
        self.model = self.load_or_create_model(model_path)

    def load_or_create_model(self, path):
        try:
            model = load_model(path)
            print("Loaded existing model @", path)
            return model
        except:
            print("Creating and training new model @", path)
            model = self.create_model()
            self.train(model)
            tfjs.converters.save_keras_model(model, 'modelfile')
            model.save(path)
            return model
        
    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv2d_1'),
            MaxPooling2D((2, 2), name='maxpool_1'),
            Dropout(0.1, name='dropout_1'),
            Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
            MaxPooling2D((2, 2), name='maxpool_2'),
            Dropout(0.1, name='droput_2'),
            Conv2D(64, (3, 3), activation='relu', name='conv2d_3'),
            Flatten(name='flatten'),
            Dense(64, activation='relu', name='dense_1'),
            Dropout(0.05, name='dropout_3'),
            Dense(47, activation='softmax', name='dense_2')  # balanced EMNIST has 47 characters
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self, model, epochs=5):
        x_train, y_train = extract_training_samples('balanced')
        x_test, y_test = extract_test_samples('balanced')
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    
    def preprocess_letter(self, letter_points):
        if len(letter_points) < 2:
            return None
            
        min_x = min(p[0] for p in letter_points)
        max_x = max(p[0] for p in letter_points)
        min_y = min(p[1] for p in letter_points)
        max_y = max(p[1] for p in letter_points)
        
        padding = 20
        width = max(max_x - min_x + 2*padding, 1)
        height = max(max_y - min_y + 2*padding, 1)
        
        # create padded image
        letter_img = np.zeros((height, width), dtype=np.uint8)
        adjusted_points = [(int(x - min_x + padding), int(y - min_y + padding)) for x, y in letter_points]
        
        # draw points as lines onto image
        for i in range(len(adjusted_points) - 1):
            cv2.line(letter_img, adjusted_points[i], adjusted_points[i+1], 255, 2)
            
        # EMNIST is 28x28
        letter_img = cv2.resize(letter_img, (28, 28), interpolation=cv2.INTER_AREA)        
        letter_img = letter_img.astype('float32') / 255.0

        if self.camera_flipped:
            letter_img = cv2.flip(letter_img, 1)
        
        return letter_img.reshape(1, 28, 28, 1)
    
    def finger_states(self, hand, index_tip, index_wrist_thresh=0.25, index_middle_thresh=0.075, index_ring_thresh=0.125):
        middle_tip = hand.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        wrist = hand.landmark[self.mp_hands.HandLandmark.WRIST]
        
        index_wrist_dist = np.linalg.norm(
            np.array([index_tip.x, index_tip.y]) - 
            np.array([wrist.x, wrist.y])
        )
        
        # finger is pointing at camera if index is close to wrist in 2D plane; draw
        if index_wrist_dist < index_wrist_thresh:
            return True, False, False
        
        # now we can assume index is raised (since it is far from the wrist)
        index_middle_dist = np.linalg.norm(
            np.array([index_tip.x, index_tip.y]) - 
            np.array([middle_tip.x, middle_tip.y]))
                
        # middle not raised; draw
        if index_middle_dist > index_middle_thresh:
            return True, False, False

        index_ring_dist = np.linalg.norm(
            np.array([index_tip.x, index_tip.y]) - 
            np.array([ring_tip.x, ring_tip.y]))
        
        # midle raised but ring not; segment to next letter
        if index_ring_dist > index_ring_thresh:
            return True, True, False
        # middle and ring raised; do not draw but allow hand movement within letter
        else:
            return True, True, True
    
    def process_frame(self, dist_thresh=300, debug_show_letters=False):
        ret, frame = self.camera.read()
        if not ret:
            return False
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # TODO support 2 hands at once?
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            index_tip = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            index_raised, middle_raised, ring_raised = self.finger_states(hand, index_tip)
            
            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
            
            # only index raised; draw
            if index_raised and not middle_raised and not ring_raised:
                if self.last_point is not None:
                    dist = np.linalg.norm(np.array([x, y]) - np.array(self.last_point))
                    if dist < dist_thresh: # adjust dist_thresh to prevent possible glitches, forces slower draw speed
                        self.letter_points[-1].append((x, y))
                        cv2.line(frame, self.last_point, (x, y), (0, 255, 0), 2)
                self.last_point = (x, y)
            
            # TODO would be cool if letters were automatically segmented; even an NN probably cannot achieve this easily, but maybe one w/ a calibration process?
            # only index and middle raised; segment letter
            # or press space
            elif index_raised and middle_raised and not ring_raised or cv2.waitKey(1) & 0xFF == 32:
                if self.letter_points[-1]:
                    letter_img = self.preprocess_letter(self.letter_points[-1])
                    if letter_img is not None:
                        if debug_show_letters:
                            cv2.imshow('Letter', cv2.resize(letter_img[0], (100, 100)))
                        pred = self.model.predict(letter_img, verbose=0)
                        pred_char = self.char_map.get(np.argmax(pred), '???')
                        print(f"Detected: {pred_char}")
                        # TODO letter error correction via whole-word prediction (basically, spell check)
                    self.letter_points = [[]]
                self.last_point = None
            
            # three raised; "pen up"
            elif index_raised and middle_raised and ring_raised:
                self.last_point = None
            
            # draw current letter
            if self.letter_points[-1]:
                points = self.letter_points[-1]
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i+1], (0, 255, 0), 2)
        
        cv2.imshow('Scribble', cv2.flip(frame, 1))
        return True

    def run(self):
        print("ESC to quit")
        print("Index finger up to write")
        print("Index and middle up to go to next letter, or press SPACE")
        print("Index, middle, and ring up to move freely without writing")
        while True:
            if not self.process_frame():
                break
                
            if cv2.waitKey(1) & 0xFF == 27:  # escape to exit
                break
                
        self.camera.release()
        cv2.destroyAllWindows()
        
# TODO command line args
if __name__ == "__main__":
    app = FingerPen(
        camera_index=0,
        model_path='air_writing_model.h5',
        mapping_path='emnist_balanced_mapping.txt'
    )
    app.run()