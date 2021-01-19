import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import *

# Bu kısımda data ve labels adında verilerimizi tutmak için listeleri oluşturuyoruz, 43 sınıf olduğunu belirtiyoruz ve şu an ki konumumuzu os.getcwd() fonksiyonu ile alıyoruz!
data = []
labels = []
classes = 43
cur_path = os.getcwd() 

# Bu kısımda train, test ve validation verilerimizin miktarlarını belirliyoruz!
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

# classes e kadar yani 0 dan 43 e kadar döngü başlatıp içerisindeki resim dosyalarını okuyoruz!
for i in range(classes):
    # Klasörün path ini alıyoruz!
    path = os.path.join(cur_path,'gtsrb',str(i))
    images = os.listdir(path)

    for a in images:
        # Klasörün içerisinde gezinip her fotoğrafı array şeklinde okuyup data ve labels listelerimize atıyoruz!
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

data = np.array(data)
labels = np.array(labels)

# Train ve Test datalarımızı oluşturuyoruz!
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size= 1 - train_ratio, random_state=42)

# Validation ve Test datalarımızı oluşturuyoruz!
X_val, X_test, Y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

# Labellerimizi kategorik veriye dönüştürüyoruz!
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Modelimizi oluşturup Conv2D, MaxPool, Dropout, Flatten ve Dense katmanlarımızı ekliyoruz!
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Modelimizi compile ediyoruz!
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Gerekli callbackleri modele eklemek için liste içerisinde bunları belirtiyoruz!
my_callbacks = [
    EarlyStopping(monitor='val_accuracy'),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', mode='min', verbose=1),
    TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True,
    write_images=False, update_freq='epoch',
    embeddings_freq=0,)]

# Epoch 15 olarak belirliyoruz!
epochs = 15

# Modeli train ve test verileriyle fit ediyoruz ayreten oluşturduğumuz callbackleri de ekliyoruz!
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test), callbacks=my_callbacks)
model.save("traffic_classifier.h5")

# Accuarcy plotlarımızı çizdiriyoruz!
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Loss plotlarımızı çizdiriyoruz!
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
