from TestingEnvironment import Environment
import random
from tensorflow import keras 
import numpy as np
import sys
import matplotlib.pyplot as plt 
env = Environment(30, 30, 30)

def act(model, state):
    better_state = np.reshape(state, (1, 30, 30, 3))
    act_values = model.predict(better_state)
    act_values = act_values[0]
    print(act_values)
    image = []
    for i in range(0, 900, 30):
        image.append(act_values[i:i+30])
    print(image)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    #return np.argmax(act_values[0])

model = keras.models.load_model("model.h5")
env.reset()
act(model, env.state)

