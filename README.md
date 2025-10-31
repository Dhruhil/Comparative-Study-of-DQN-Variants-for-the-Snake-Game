# Comparative Study of DQN Variants for the Snake Game

** Overview:
A comparative study of Vanilla DQN, Double DQN, and Dueling DQN was conducted to develop an AI agent capable of learning optimal policies in a 20×20 Snake game environment. Leveraging OpenAI Gym, a structured state space (Empty, Snake Body, Food), discrete action set (Forward, Turn Left, Turn Right), and a reward function were defined to promote efficient food collection and safe navigation. The agents were trained using ε-greedy exploration, experience replay, and target network updates, with systematic hyperparameter tuning including learning rate, discount factor, and batch size. Over 100 evaluation episodes, Dueling DQN demonstrated superior performance, achieving the highest mean reward (357.7), score (21.93), and lowest variance, surpassing Vanilla and Double DQN in learning stability, efficiency, and generalization. This study highlights the potential of advanced reinforcement learning architectures and methodical hyperparameter optimization to enhance autonomous decision-making in high-dimensional sequential environments.

## Before We Start

Make sure that you've installed all required modules:

```sh
pip3 install -r requirements.txt
```

## Run the Scripts

To begin with, run the script with this command to take a look at our (maybe)
best agent:

```sh
python3 evaluate_agents.py models/snake_parse_1/02-dueling
# OR
./evaluate_agents.py models/snake_parse_1/02-dueling   # Unix only
```

To train the agents, simply run:

```sh
python3 train.py
```

This trains all models concurrently.

## File Structure

This repository consists of the following:

* `config`: Configuration files.
  * `config.json`: Configuration file for training, with hyperparameters and
      reward values. _Note:_ If you wish to control what agents should be trained,
      add or remove configurations in the `agents` array and override
      hyperparameters and reward values.
* `models`: Trained models. `train.py` also stores the models here.
  * `evaluation_results_agent.{csv,png}`: The evaluation metrics and visualization of
    different agent types.
  * `evaluation_results_tuning.{csv,png}`: The evaluation metrics and visualization of
    different hyperparameters and rewards.
  * Content of subdirectories (`00-vanilla`, etc.):
    * `agent_config.json`: Agent name, hyperparameters, and rewards.
    * `best_model.pth`: The model that has best Mean(10) value.
    * `checkpoint_*.pth`: Checkpoint models.
    * `final_model.pth`: The final model file, to be evaluated.
    * `training_curves.png`: Episode rewards and training loss for each episode.
    * `training-log.csv`: More detailed chart containing Epsilon, Reward, Avg10 and Loss.
* `src`: The main logic of the training.
  * `agents/*_DQN.py`: Define the DQN Agent classes.
  * `envs/advanced_snake_env.py`: Implements the environment, including reward logic.
  * `models/advanced_networks.py`: Implements deep Q-Network support for both
    vector (MLP) and grid (CNN) inputs.
  * `training/trainer.py`: Implements the training loop and metric logging.
* `evaluate_agents.py`: Evaluate the trained agents and output the results.
* `train.py`: The main entry for training.
* `watch_snake.py`: Displays the animated snake game using the given agent.
  (It requires the base path of the model as the sole command-line argument)
