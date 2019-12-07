# Industrial_control
Testing Reinforcement Learning (**RL**) methods on industrial control problems using a Deep Q-Network (**DQN**) agent within a simulation environment mimicking a reagent addition procedure in a simple batch process setting. 

## Code

### Running
```
python DQN_reagent_addition_100x100_Sim_static_goal.py
```

### Dependencies

*  CV2
*  Numpy
*  Tensorflow
*  Keras
*  Pillow


## Detailed Description
### Aim and Environment

This projects aim is to probe the application of RL agents to industrial control problems, starting with a simple reagent addition to achieve a desired final mixture volume and quality attribute (colour in this example). 

The environment shown below is a 10x10 or a 100x100 RGB matrix  depicting source tanks (starting reagents, 2 to 4) in the top row and destination tank (starts empty, lower left) and goal destination tank (lower right) that shows the desired final mixture volume and colour.   

**Environment 10x10 at step = 2**

<img src="https://github.com/jameshiggie/Industrial_control/blob/master/img/10x10_start_anno.png" width="400">

An action is a selection of one source tank to transfer one unit of reagent to the destination tank, therefore the size of the action space is equal to the number of source tanks. The end state is researched when the volume and colour of the destination tank matches the goal destination tank. At first the end state was experimented to be +/- 10% of requirements to effectively explore hyper parameter affects. A negative reward of 1 is given for each addition of reagent to optimise reagent usage. A large negative reward of 300 is given if the destination tank is overfilled. A positive reward of 100 is given at the achievement of the end state, correct volume and colour of mixture within the destination tank. During experimentation the reward function was set to be a positive function of how close the colour was to the goal colour, however it lead to the agent getting stuck in the early stages of training more often than the simpler rewards as previously stated. Below is an example of a run where the agent reached the completion state.

**Environment 10x10 at end point**

![Environment 10x10 at end state](https://github.com/jameshiggie/Industrial_control/blob/master/img/10x10_end.png)

---

### DQN Agent

This DQN agent is made of two (self and target) Convolutional Neural Networks (**CNN**) to approxiamte Q values. The Q values approximate the expected return for any action at any state. Each CNN consists of two convolutional layers (size of 128 (3,3) and 64 (3,3) respectively) with relu activation, (2,2) maxpooling and 20% to 10% drop out respectively followed by two dense layers (size of 32 and "action space size" respectively) with a linear activation.  

#### Training

The loss function used in training is calculated as the reward + discount * max_future_q

![Loss_Function](https://github.com/jameshiggie/Industrial_control/blob/master/img/loss_func.png)


## Results and discussion

This method works well for our 10x10 and 100x100 environments with a static goal and only two source tanks, below are the tensor board results for each environment solved respectively. We have not yet determined how solve the 100x100 environment with 4 source tanks (all primary regents required to achieve any colour) and a random volume and colour goal destination tank and does not work with the static goal agents. 

**10x10 static goal results** 
![10x10_tb](https://github.com/jameshiggie/Industrial_control/blob/master/img/tb1.png)

**100x100 static goal results** 
![100x100_tb](https://github.com/jameshiggie/Industrial_control/blob/master/img/tb2.png)

## Resources and links
* ![OG_Paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) - DeepMinds original paper on DQN: Human-level control through deep reinforcement learning

* ![SentDex DQN Tutorial](https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial) - Initial learning material used from Sentdex and basis of Keras model


## License
This project is licensed under the MIT License, see the [LICENSE.md](LICENSE.md) file for details
