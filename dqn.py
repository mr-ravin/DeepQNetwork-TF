import tensorflow as tf
import numpy as np
import random
import gym
BATCH_SIZE=10
MAX_EPSILON=0.7
MIN_EPSILON=0.1
LAMBDA=0.1
GAMMA=0.1

### functions on replay buffer ###
def add_experience(lst,max_memory,new_exp): ## add experience to replay buffer
  lst.append(new_exp)
  if len(lst)> max_memory:
    lst.pop(0)
  return lst

def get_prev_experience(lst,number_of_samples): ## retrieve sample from replay buffer
  if number_of_samples> len(lst):
    return random.sample(lst,len(lst))
  else:
    return random.sample(lst,len(number_of_samples))
### functions on replay buffer ###


def train(batch_size,render=True): ## method to train 
  env_name='MountainCar-v0'
  max_memory=5000
  total_reward=0
  max_distance=-100
  lst=[]
  max_x_store=[]
  reward_store=[]
  env=gym.make(env_name)
  num_states=env.env.observation_space.shape[0]
  num_actions=env.env.action_space.n

 with tf.Session() as sess:
  ### building graph
  states=tf.placeholder(shape=[None,num_states],dtype=tf.float32)
  Qsa=tf.placeholder(shape=[None,num_actions],dtype=tf.float32)
  fc1=tf.layers.dense(states,30,activation=tf.nn.relu)
  fc2=tf.layers.dense(fc1,30,activation=tf.nn.relu)
  logits=tf.layers.dense(fc2,num_actions)
  loss=tf.losses.mean_squared_error(Qsa,logits)
  opt=tf.train.AdamOptimizer().minimize(loss)
  ### building graph
  sess.run(tf.global_variables_initializer) ### initialize the neural network variables
