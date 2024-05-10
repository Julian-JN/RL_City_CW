# RL_City_CW
Reinforcement Learning Coursework.

For tabular Q-Learning (Task 1), the main modules are `maze_env.py` and `agent.py` which describe the environment and the Q-Learning algorithm respectively. To train the Q-learning agent, run `agent.py` file. To change the parameters of environment and algorithm, change them in line 255 and 257 of this file. There are two reward-transition modes for the environment ("limited_movement" and "terminal_movement" parameters). Similarly, there are two policy modes ("greedy" and "softmax")

For Deep Q-Learning, DQN (Advanced Task) and extensions, the main modules are `dqn.py` which describes the neural networks and trains it (saving the checkpoints) and `dqn_inference.py` which loads a checkpoint from the `rl_chk` directory and generates a video. Note that the videos have already been uploaded. The hyperparameters can be changed in the `__init__()` method of `dqn.py`. Control over which variant of DQN to use (DQN, DDQN and PER) can be controlled in the main function where the agent is instantiated. To change the checkpoint for inference, change the name of the loaded checkpoint in the line `checkpoint = torch.load(f"rl_chk/double-dqn-checkpoint_4mil.pth")`. Note that the videos have been loaded in the directory `videos`.

The repository has two branches, showcasing the individual parts of the coursework with different Atari environments.
