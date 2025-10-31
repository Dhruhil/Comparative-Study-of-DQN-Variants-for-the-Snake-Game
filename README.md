# Comparative Study of DQN Variants for the Snake Game

A **comparative study** of **Vanilla DQN**, **Double DQN**, and **Dueling DQN** was conducted to develop an **AI agent** capable of learning optimal policies in a **20×20 Snake game environment**.

Using **OpenAI Gym**, the environment was designed with:
- **State space:** Empty, Snake Body, and Food  
- **Action space:** Forward, Turn Left, and Turn Right  
- **Reward function:** Encourages efficient food collection and safe navigation  

---

### Training Methodology

The agents were trained using the following reinforcement learning techniques:

- **ε-greedy exploration** for balancing exploration and exploitation  
- **Experience replay** to improve sample efficiency  
- **Target network updates** for training stability  
- **Systematic hyperparameter tuning**, including:
  - Learning rate  
  - Discount factor (γ)  
  - Batch size  

---

###  Evaluation

Over **100 evaluation episodes**, **Dueling DQN** demonstrated the **best performance**, achieving:
- **Highest mean reward:** 357.7  
- **Highest mean score:** 21.93  
- **Lowest variance** among all models  

It surpassed both **Vanilla DQN** and **Double DQN** in terms of **learning stability**, **training efficiency**, and **generalization**.

---

###  Key Insight

This study highlights the **impact of architectural advancements** in Deep Q-Learning and **methodical hyperparameter optimization** on improving **autonomous decision-making** within **high-dimensional sequential environments**.

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
