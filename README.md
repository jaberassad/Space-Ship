# Lunar Lander Deep Q-Learning Project

## Overview
This project uses Deep Q-Learning to train an agent to land a lunar spaceship in the **LunarLander-v2** environment from OpenAI's Gym. The agent uses a neural network to estimate the optimal actions, while the training is based on the experience replay technique and the epsilon-greedy policy.

## Dependencies
Make sure you have the following dependencies installed before running the project:
- `numpy`
- `tensorflow`
- `gymnasium`
- `bayes_opt`
- `python 3.11`
- `Git`

You can install them using pip:
```bash
pip install numpy tensorflow gymnasium bayesian-optimization
```

## Running the Project
### 1.Cloning the repository
```bash
git clone https://github.com/jaberassad/Space-Ship.git
cd Space-Ship
```


###  2.Running the Agent
After cloning, you can test the agent's performance by running the following command. The trained model will be loaded, and the agent will attempt to land the spaceship in a visual environment.

```bash
python test_agent.py
```

## Training Approach
The project uses **Deep Q-Learning** to solve the lunar landing task. Here’s a brief summary of the approach:
- **Neural Network**: The agent uses a fully connected neural network for approximating the Q-value function. The network's architecture is determined by hyperparameters such as the number of layers and neurons per layer.
- **Epsilon-Greedy Policy**: The agent selects actions based on an epsilon-greedy policy, where it chooses a random action with probability epsilon and the action with the highest Q-value otherwise. Over time, epsilon decays to encourage exploitation of the learned policy.
- **Experience Replay**: The agent stores its experiences (state, action, reward, next state) in a memory buffer and samples batches from this memory to train the network, which helps break correlation between consecutive experiences.
- **Discounted Future Rewards**: The agent updates the Q-values based on both the immediate reward and the discounted estimated future reward, encouraging long-term success.

### Hyperparameter Tuning
Bayesian Optimization was used to tune the hyperparameters, including:
- **Learning rate**
- **Number of layers and neurons**
- **Discount rate**
- **Epsilon decay**
- **Batch size**

The optimizer maximizes the agent's average reward over the first 100 episodes to find the best combination of these parameters.

### Example Code Snippet for Training
```python
space_ship = Agent(
    batch_size=179, 
    discount_rate=0.99, 
    epsilon_decay=0.995, 
    learning_rate=0.001346, 
    num_layers=3, 
    num_neurons=[128, 128, 0, 0, 0], 
    eps=200
)
space_ship.run()
```

## Files
- `train_agent.py`: Contains the main code for training the lunar lander agent.
- `test_agent.py`: Used to test the agent after training by loading the saved model.
- `bayesian_optimization.py`: Optional code for hyperparameter tuning using Bayesian optimization.

## Future Improvements
- **Better Neural Network Architecture**: Explore more advanced architectures like convolutional layers or recurrent networks.
- **More Environments**: Extend the training to other Gym environments or custom environments.
- **Transfer Learning**: Train the agent on similar tasks to leverage previously acquired knowledge.
