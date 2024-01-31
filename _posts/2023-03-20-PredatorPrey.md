---
layout: post
title: "Predator Prey Reinforcement Learning"
date: 2023-03-20T00:00:00-00:00
author: Aman Priyanshu
categories: technical-blogs
cover: "/assets/demo.gif"
---

## Introduction

In this blog post, I'll introduce the concept of Reinforcement Learning (RL) Systems, Multi-Agent Reinforcement Learning (MARL) Systems, and demonstrating a project based on them. However, before we begin discussing these concepts in detail, we can begin by understanding the motivations behind reinforcement learning. Humans have the ability to learn complex decision making processes fairly easily. We've the ability to solve long-form complex puzzles and conditional games, however, it becomes difficult to do the same with computational systems. The idea behind teaching computers to learn based on rewards and simulations instead of supervised/unsupervised learning makes the backbone of RL. These systems are powerful tools for tackling complex decision-making problems. They are used to create AI agents that can learn from their environment and take actions to maximize their rewards. For example, when considering driving a car, we realize that humans have the ability to drive them efficiently and safely, however it is a difficult task for a computer to learn. RL systems can be used to create an AI agent that can learn to drive a car by taking actions in a simulated environment and receiving rewards for its actions.

My project can be found [here](https://github.com/AmanPriyanshu/PredatorPreyRL).

![Demo](https://chaiwithpy.github.io/assets/demo.gif)

## Preliminaries

### Reinforcement Learning

![Reinforcement Learning](https://chaiwithpy.github.io/assets/Reinforcement_learning_diagram.svg)

Reinforcement Learning (RL) is a type of Machine Learning in which an agent learns to interact with its environment by taking actions and receiving rewards. The agent learns to maximize its rewards by learning a policy that maps states to actions. While there are a multitude of ways to go about executing RL, model-free RL has become popular primarily due to its use of a Deep Neural Network to map states and actions directly. This allows learning over complex environments and employs the predictive capability of DNNs, thus enabling better performance without a complexity tradeoff.

#### Model-Free RL

Here, we specifically focus on model-free RL, where the agent interacts with the environment and receives a reward and the next observation. These interactions and actions are extracted by mapping the current observation and maybe the last few to the best action to take without explicitly modeling what happens in the environment through a neural network. This is usually done by training a neural network to take in a state and output an action. Policy gradient methods are often used for this, as they allow for backpropagation even though the reward isn't backpropagatable. 

$$
Q(s,\ a)=\ r(s,\ a)\ +\ \gamma\max_a {Q(s',\ a)}
$$

This allows the model to augment its learning without needing a model for the entire world. Thus creating a model-free RL system.

### Multi-Agent RL (MARL)

Multi-Agent RL is an extension of RL in which multiple agents interact with each other and their environment simultaneously. This enables more complex and real-world like simulations, as well as the ability for agents to learn from each other's experiences. This type of RL can be used to solve a range of tasks, from robotics to game playing.

This type of RL can be used to solve a range of tasks, from robotics to game playing. For example, within games, multiple agents can be used to play against each other, allowing for more complex strategies to be learnt. Additionally, MARL systems can also be used to solve problems in areas such as finance and healthcare. For example, in finance, multiple agents can be used to simulate the stock market, allowing for more accurate predictions.

Overall, Multi-Agent RL is an incredibly powerful tool for solving complex tasks, as it allows for agents to learn from each other's experiences and interact with their environment in a more realistic way. I hope to explore this specific paradigm of RL through this blog.

## Implementation:

This section provides an overview of the implementation of Multi-Agent Reinforcement Learning (MARL) for the predator-prey task. The goal of this task is to have a group of agents (predators) learn to capture a single prey agent in a two-dimensional environment. The environment is a grid-world with different density spaces, specifically represented through the green grasses and the blue waters. The environment is specifically designed to allow the prey to run faster, thus allowing the predators to demonstrate strategies such flanking for capture.

The MARL implementation uses a centralized training approach, where the predator agents share a common policy and base neural network. The agents are trained using a Deep Q-Network (DQN) with experience replay. All agents are trained using a combination of exploration and exploitation. The exploration is done using an epsilon-greedy policy, where the agents explore the environment by randomly selecting actions with a certain probability. The exploitation is done by selecting the action with the highest expected reward.

### Environment Generation

The environment is generated to simulate different density spaces for the agents to traverse over, therefore Pelin noise was used to create a noisy spectrum of nature-like landscapes. 

```py
self.noise1 = PerlinNoise(octaves=3, seed=self.seed)
self.noise2 = PerlinNoise(octaves=6, seed=self.seed)
self.noise3 = PerlinNoise(octaves=12, seed=self.seed)
for i in range(self.h):
    row = []
    for j in range(self.w):
        arr = [i/self.h, j/self.w]
        noise_val = self.noise1(arr)
        noise_val += 0.5 * self.noise2(arr)
        noise_val += 0.25 * self.noise3(arr)
        row.append(noise_val)
    self.basemap.append(row)
self.basemap = np.array(self.basemap)
```

Also to ensure sharp features, these vectors were polarized in depth through:
```py
self.basemap[self.basemap<0.5] = np.power(self.basemap[self.basemap<0.5], 2)
```


### Prey Model

```py
class PreyModel(torch.nn.Module):
    def __init__(self):
        super(PreyModel, self).__init__()
        self.linear1 = torch.nn.Linear(5, 16)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 4)

    def forward(self, state):
        x = self.linear1(state)
        x = self.activation(x)
        x = self.linear2(x)
        x_ = self.activation(x)
        x = self.linear3(x_)
        return x
```

The aim of the `PreyModel` is to effecitively evade the predators, here the inputs are its own location `x, y`, the relative closest location of the nearest predator `pred_x - x, pred_y - y`, and the `density` of the environment. These attributes allow the agent to effectively navigate the environment while evading the predators.

### Predator Model

```py
class PredatorModel(torch.nn.Module):
    def __init__(self):
        super(PredatorModel, self).__init__()
        self.linear1 = torch.nn.Linear(13, 16)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 4)

    def forward(self, state):
        x = self.linear1(state)
        x = self.activation(x)
        x = self.linear2(x)
        x_ = self.activation(x)
        x = self.linear3(x_)
        return x, x_
```

Here, the aim of the model is to extract the linear rewards associated with each of the 4 directions of movement. The input is described as the current positions `x, y` of the agent, `prey_x - x, prey_y - y` relative positioning of the prey, and the `density` of the environment, the `communication_embeddings` which are extracted from all agents as follows:

```py
communications = []
for pred in self.preds:
    scream = pred.scream([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]])
    communications.append(scream)
communications = np.array(communications)
communications = np.mean(communications, axis=0)
```

Where, `pred.scream` is the extracted vector from transition `linear2` layer.

## Usage

The code can be used to train agents from scratch using the `env.py` script as follows:

```py
if __name__ == '__main__':
    env = Environment()
    env.start()
    N = 1000
    bar = trange(N)
    loss_prey = ""
    loss_preds = ""
    for i in bar:
        if env.game_condition:
            env.reset()
        prey_reward, pred_rewards = env.take_actions()
        if i%8==0 and i>1:
            loss_prey, loss_preds = env.train_step()
            loss_prey = str(round(loss_prey, 3))
            loss_preds = str(round(loss_preds, 3))
            print()
        bar.set_description(str({"prey_reward": round(prey_reward, 3), "pred_rewards": round(np.mean(pred_rewards), 3), "loss_prey": loss_prey, "loss_preds": loss_preds}))
        bar.update()
        if i>N:
            break
    bar.close()
    torch.save(env.prey.model.state_dict(), "prey_model.pt")
    torch.save(env.predator_model.state_dict(), "pred_model.pt")
```

Test-runs can be explored through the `test.py` script as follows:

```py
if __name__ == '__main__':
    np.random.seed(5)
    env = Environment()
    env.start()
    env.prey.model.load_state_dict(torch.load("prey_model.pt"))
    env.predator_model.load_state_dict(torch.load("pred_model.pt"))
    for i in trange(50):
        env.prey.epsilon = 1.0
        for idx, pred in enumerate(env.preds):
            pred.epsilon = 1.0
        env.take_actions()
        colored_map = env.plot(False)
        colored_map = colored_map*255
        cv2.imwrite("./imgs/img_"+"0"*(2-len(str(i)))+str(i)+".png", colored_map)
```

## Conclusion

This blog discussed the predator prey model under the lens of multi-agent reinforcement learning systems. It aimed at exploring the simulation of the interactions between predators and prey in a natural ecosystem, using varied density spaces. As can be seen in the demo, the predators are capable of learning to surround the prey while pursuing it for capture. This is interesting due to its real-world correlations, especially due to the agent's abilities to communicate without knowing each others reference locations.

You can find my project [here](https://github.com/AmanPriyanshu/PredatorPreyRL).