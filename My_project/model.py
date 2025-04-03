import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, add

def create_complex_model(model_path):
    input_layer = Input(shape=(256, 256, 3))
    
    # Branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    
    # Branch 2
    branch2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv2D(64, (5, 5), activation='relu', padding='same')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
    
    # Merge branches
    merged = add([branch1, branch2])
    merged = Conv2D(128, (3, 3), activation='relu', padding='same')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = GlobalAveragePooling2D()(merged)
    
    # Fully connected layers
    fc = Dense(256, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    output_layer = Dense(3, activation='softmax')(fc)  # Assuming 3 classes for classification
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Save the model
    model.save(model_path)
    print(f"Complex model saved at: {model_path}")

# Create new directory if it doesn’t exist
model_dir = 'new_model'
os.makedirs(model_dir, exist_ok=True)

# Define model file path
model_file = os.path.join(model_dir, 'complex_model.h5')

# Create and save model
create_complex_model(model_file)
