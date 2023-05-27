import numpy as np
import tensorflow.keras.applications.vgg16
from tensorflow.keras.layers import concatenate, Dense, Dropout, Embedding, Input, LSTM, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image

# Define the maximum caption length and vocabulary size
max_len = 100
vocab_size = 10000

# Define the image input
image_input = Input(shape=(2048,))
image_model = Dropout(0.5)(image_input)
image_model = Dense(256, activation='relu')(image_model)

# Define the partial caption input
partial_captions = Input(shape=(max_len,))
caption_embedding = Embedding(vocab_size, 256, mask_zero=True)(partial_captions)
caption_embedding = Dropout(0.5)(caption_embedding)
caption_model = LSTM(256)(caption_embedding)

# Reshape the inputs for concatenation
image_model = Reshape((1, 256))(image_model)
caption_model = Reshape((1, 256))(caption_model)

# Concatenate the image and caption models
decoder = concatenate([image_model, caption_model])

# Additional layers
decoder = LSTM(256)(decoder)
decoder = Dense(256, activation='relu')(decoder)
decoder = Dropout(0.5)(decoder)

# Output layer
outputs = Dense(vocab_size, activation='softmax')(decoder)

# Create the caption model
caption_model = Model(inputs=[image_input, partial_captions], outputs=outputs)

# Compile the model
caption_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001))

# Save the model
caption_model.save('caption_model.h5')

# Load the model
caption_model = load_model('caption_model.h5')

# Function to preprocess the image
def preprocess_image(img_path, model_type):
    if model_type == 'VGG16':
        target_size = (224, 224)
        preprocess_input = vgg16_preprocess_input

    img = image.load_img(img_path, target_size=target_size)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

# Function to generate captions
def generate_captions(image_path, model_type, tokenizer, max_len, num_captions):
    img_data = preprocess_image(image_path, model_type)
    image_features = model.predict(img_data)

    captions = []
    for _ in range(num_captions):
        caption_input = np.zeros((1, max_len))

        for i in range(max_len):
            caption_input[0, i] = tokenizer.word_index['startseq']
            preds = caption_model.predict([image_features, caption_input])
            word_pred = np.argmax(preds[0, i])
            caption_input[0, i + 1] = word_pred

            if tokenizer.index_word[word_pred] == 'endseq':
                break

        caption = []
        for word_index in caption_input[0]:
            word = tokenizer.index_word[word_index]
            if word == 'startseq':
                continue
            if word == 'endseq':
                break
            caption.append(word)
        caption = ' '.join(caption)
        captions.append(caption)

    return captions

# Test the model with a sample image
image_path = 'image1.png'
model_type = 'VGG16'  # Specify the model type used for image feature extraction
tokenizer = tokenizer  # Specify the tokenizer used for caption generation
max_len = 100  # Maximum caption length
num_captions = 5  # Number of captions to generate

captions = generate_captions(image_path, model_type, tokenizer, max_len, num_captions)
for caption in captions:
    print(caption)
