import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc
# Enable memory growth to avoid GPU memory issues
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    print("No GPU devices found or unable to set memory growth")
class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[-1],),
                                initializer='zeros',
                                trainable=True)
        
    def call(self, inputs):
        score = tf.matmul(inputs, self.W) + self.b
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        return context_vector
def create_model(input_shape, num_classes):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = inputs / 255.0
    
    x = base_model(x)
    x = layers.Reshape((1, -1))(x)
    x = AttentionLayer()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=10, target_size=(224, 224)): # batch_size=32
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.indexes = np.arange(len(self.image_paths))
        
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        batch_images = []
        for path in batch_paths:
            try:
                image = cv2.imread(path)
                image = cv2.resize(image, self.target_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                batch_images.append(image)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                batch_images.append(np.zeros((*self.target_size, 3)))
                
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)
def get_image_paths_and_labels(dataset_dir):
    image_paths = []
    labels = []
    
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(label)
    
    return np.array(image_paths), np.array(labels)
def train_model():
    try:
        # Parameters
        INPUT_SHAPE = (224, 224, 3)
        BATCH_SIZE = 16
        EPOCHS = 20
        DATASET_DIR = "D:\IIT\Subjects\(4605)IRP\Devlo\Augmented_DataSet"
        
        print("Getting image paths and labels...")
        image_paths, labels = get_image_paths_and_labels(DATASET_DIR)
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        X_train_paths, X_test_paths, y_train, y_test = train_test_split(
            image_paths, y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_encoded
        )
        
        train_generator = DataGenerator(X_train_paths, y_train, batch_size=BATCH_SIZE)
        test_generator = DataGenerator(X_test_paths, y_test, batch_size=BATCH_SIZE)
        
        print("Creating and compiling model...")
        model = create_model(INPUT_SHAPE, num_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        print("Training model...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=test_generator,
            callbacks=callbacks
        )
        
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        
        model.save('tree_classifier_attention.keras')
        
        return model, history, label_encoder
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

try:
    model, history, label_encoder = train_model()
except Exception as e:
    print(f"Training failed: {str(e)}")
