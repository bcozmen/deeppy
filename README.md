# deeppy
A deep learning framework that separates network architecture, data and the algorithm independent of each-other for flexible training.  

1) Create a dataset object to handle the preprocessed dataset for supervised or unsupervised tasks or environment for the RL algorithms
Renders as:
```python
env = gym.make("CartPole-v1", render_mode="rgb_array")
data = EnvData(env, 20000, device= device)
```  
2) Decide on a network architecture, using torch.nn
```python
q_arch_params = {
    "layers" : [4,128,128,2],
    "type" : nn.Linear,
    "hidden_act" : nn.ReLU,
    "out_act" : nn.Identity,
    "weight_init" : "uniform"
}
``` 
3) Choose an algorithm to train
```python
model = Discrete_SAC(**sac_params)
``` 
4) And lastly, initialize and train your learning frame
```python
lf = LearningFrame(model, data, batch_size = 128, start_size = 128)

#For each epoch
for epoch in range(EPOCH):
	#Play one game
	while(not lf.train()):
            pass
lf.plot(lf.rewards,show_result= True)
``` 

For example usage see [this jupyter notebook](examply.ipynb)

# Currently Implemented Algorithms

[DQN](models/rl/dqn.py)

<details>
  <summary>Details</summary>
</details> 

[Double DQN](models/rl/dqn.py)


<details>
  <summary>Details</summary>
</details> 

[Clipped Double DQN](models/rl/dqn.py)


<details>
  <summary>Details</summary>
</details> 

[Discrete SAC](models/rl/sac.py)

<details>
  <summary>Details</summary>
</details> 

[Regression, Classification, Auto-encoders](networks/network.py) 


<details>
  <summary>Details</summary>
</details> 

# To do
### Policies

[Beta-VAE]

[Dueling DQN]

[PPO]

[Model Based Policy Optimization (MBPO)]

[SafeMBPO]

### Features
Save-load model checkpoints

