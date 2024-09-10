import numpy as np
import gym 
import tensorflow as tf


env = gym.make("LunarLander-v2", render_mode="human")
model = tf.keras.models.load_model("model.keras")
for episode in range(10):
    state = env.reset()
    done = False
    if isinstance(state, tuple):
        state = np.array(state[0])
    state=state.reshape((1,8))

    score=0
    while not done:
        action = np.argmax(model.predict(state, verbose=0))
        next_state, reward, done, info, _ = env.step(action)
        if isinstance(next_state, tuple):
            next_state = np.array(next_state[0])
        next_state=next_state.reshape((1,8))
        state = next_state
        score+=reward

        if done or (state[0][6]==1 and state[0][7]==1):
            print("episode: {}/{}, score:{}".format(episode+1, 10, score))
            break
