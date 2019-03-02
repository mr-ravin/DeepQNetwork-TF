import tensorflow as tf
import numpy as np
import math
import random
import gym
import matplotlib.pyplot as plt
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
    return random.sample(lst,number_of_samples)
### functions on replay buffer ###


def train(batch_size,game_name="MountainCar-v0",render=True): ## method to train 
 max_memory=5000
 num_episodes=300 ### Number of epochs
 lst=[]           ### List containing <state,action,reward,next_state>
 max_x_store=[]
 reward_store=[]
 ####Start: Environment related details
 env_name=game_name
 env=gym.make(env_name)
 num_states=env.env.observation_space.shape[0]
 num_actions=env.env.action_space.n
 ####End: Environment related details

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
  sess.run(tf.global_variables_initializer()) ### initialize the neural network variables
  
  cnt=0
  while cnt<num_episodes:
    if cnt %10==0:
      print('Episode {} / {}'.format(cnt+1, num_episodes))
    
    #----###Start: Adding experience to replay buffer
    replay_state=env.reset()
    total_reward=0
    max_x=0
    eps=MAX_EPSILON
    steps=0
    while True:
      if render:
        env.render()
      ####Start: select action based on random threshold
      comp_val=random.random()
      if comp_val<eps:
        replay_action=random.randint(0,num_actions-1)
      if comp_val>=eps:
        replay_action=np.argmax(sess.run(logits,feed_dict={states:replay_state.reshape(1,num_states)}))
      ####End: select action based on random threshold
      replay_next_state,replay_reward,replay_done,info=env.step(replay_action)

      ####Start: Reward allocation system
      if replay_next_state[0]>=0.1:
        replay_reward=replay_reward+10
      if replay_next_state[0]>=0.25:
        replay_reward=replay_reward+20
      if replay_next_state[0]>=0.5:
        replay_reward=replay_reward+100
      ####End: Reward allocation system

      if replay_next_state[0]>max_x:
        max_x=replay_next_state[0]
      if replay_done:
        replay_next_state=None
      lst=add_experience(lst,max_memory,(replay_state,replay_action,replay_reward,replay_next_state))
      #----###End: Adding experience to replay buffer

      ###----#Start: Training using replay buffer
      train_batch=get_prev_experience(lst,BATCH_SIZE)
      train_states=np.array([val[0] for val in train_batch])
      train_next_states=np.array([(np.zeros(num_states) if val[3] is None else val[3]) for val in train_batch])
      train_qsa=sess.run(logits, feed_dict={states: train_states})
      train_qsad=sess.run(logits,feed_dict={states:train_next_states})
      train_x=np.zeros((len(train_batch),num_states))
      train_y=np.zeros((len(train_batch),num_actions))
      for idx,b in enumerate(train_batch):
        tmp_state,tmp_action,tmp_reward,tmp_next_state=b[0],b[1],b[2],b[3]
        current_q=train_qsa[idx]
        if tmp_next_state is None:
          current_q[tmp_action]=tmp_reward
        else:
          current_q[tmp_action]=tmp_reward+GAMMA*np.amax(train_qsad[idx]) # Q(s,a) = r + gamma * max(Q(s',a'))
        train_x[idx]=tmp_state
        train_y[idx]=current_q
      sess.run(opt,feed_dict={states:train_x,Qsa:train_y}) ### TRAINING DONE HERE !!!
      ###----#End; Training using replay buffer

      steps=steps+1
      eps=MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)
      replay_state=replay_next_state
      total_reward=total_reward+replay_reward
      if replay_done:
        reward_store.append(total_reward)
        max_x_store.append(max_x)
        break     #### BREAK WHILE LOOP HERE
    print("Step {}, Total reward: {}, Eps: {}".format(steps, total_reward, eps))
    cnt=cnt+1

  ####Start: plot visual graph
  plt.plot(reward_store)
  plt.show()
  plt.close("all")
  plt.plot(max_x_store)
  plt.show()
  #####End: plot visual graph
train(BATCH_SIZE)
