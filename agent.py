import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import *
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class Agent(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, target_update_interval,
                 hidden_size, learning_rate, exploration_scaling_factor):
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")

        self.critic = Critic(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)
        
        self.critic_target = Critic(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        hard_update(self.critic_target, self.critic) #TODO write hard update function.

        self.policy = Actor(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory : ReplayBuffer, batch_size, updates):
        
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        # Predictive model here. 
        # TODO: Add a predictive model. 

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Update the critic network. 
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def train(self, env, env_name, memory : ReplayBuffer, episodes=1000, batch_size=64, updates_per_step=1, summary_writer_name="", max_episode_steps=100):

        warmup = 20

        # Tensorboard
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        # Training Loop
        total_numsteps = 0
        updates = 0

        for i_episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, _ = env.reset()

            while not done and episode_steps < max_episode_steps:
                if warmup > i_episode:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)
                
                if memory.can_sample(batch_size=batch_size):
                    for i in range(updates_per_step):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.update_parameters(memory, 
                                                                                                            batch_size, 
                                                                                                            updates)
                        
                        # Tensorboard
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy', ent_loss, updates)
                        writer.add_scalar('parameters/alpha', alpha, updates)
                
                next_state, reward, done, _, _ = env.step(action)

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon
                mask = 1 if episode_steps == max_episode_steps else float(not done)

                memory.store_transition(state, action, reward, next_state, mask)

                state = next_state
            
            writer.add_scalar('reward/train', episode_reward, i_episode)
            print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")

            if i_episode % 10 == 0:
                self.save_checkpoint()



    def save_checkpoint(self):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print('Saving models')
        self.policy.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoint(self, evaluate=False):

        try:
            print('Loading models...')
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            print('Successfully loaded models')
        except:
            if(evaluate):
                raise Exception("Unable to evaluate models without a loaded checkpoint")
            else:
                print("Unable to load models. Starting from scratch")


        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()

        