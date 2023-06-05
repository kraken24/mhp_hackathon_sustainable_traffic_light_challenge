##### Original Source: https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control
*Please follow the tutorial guidelines on his github repository.*

<img src="https://www.dtekcustoms.com/wp-content/uploads/2021/02/s2.jpg" alt="drawing" width="350"/>
We have tweaked the underlying REINFORCEMENT LEARNING algorithm to improve the reliability and convergence of the neural network (RL Agent). We have used two neural networks instead of one neural network. This strategy is commonly known as using fixed weights or use of target network. It is one of the important stratgies to counter the DEADLY TRIAD in REINFORCEMENT LEARNING.

Use of Target Neural Network: In this methodology, a target neural network, identical to the neural network of the agent is used to estimate the expected action value. The temporal difference [TD(0)] learning equation used for training the agent (or in other words, updating the weights of the neural network) changes:

Bellman Equation: *Q(s,a) = reward + gamma • max Q'(s',a')* (for reference)

Original Equation: *Q(s,a) = [1 - lr] * Q(s,a; the) + lr *[reward + gamma • max Q'(s',a'; theta)*]

New Equation: *Q(s,a) = [1 - lr] * Q(s,a; the) + lr *[reward + gamma • max Q'(s',a'; phi)*]

where:
* the: parameters of Action Network (RL Agent)
* phi: parameters of Target Network
* lr: learning rate

The weights of Action Network are copied to Target Network after every 100 epochs.

Files with suffix "2" are new files, created by the team for this project
* Most of the remaining files have been used as they were written by the author

### Team: Reinforced Traffic Dejammers
<img src="https://user-images.githubusercontent.com/55467370/156913450-79379898-4cfd-4b1e-9e8e-5ec206d00d98.PNG" alt="drawing" width="550"/>
