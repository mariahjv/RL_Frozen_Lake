 # SARSA Process
     
class Sarsa_agent(object):
      def sarsa_agent(self, alpha, gamma, epsilon, n_episodes): 
            """
            alpha: learning rate
            gamma: exploration parameter
            n_episodes: number of episodes
            """
            # initialize Q table
            Q = self.init_q(self.n_states, self.n_actions)
            # initialize processing bar
            t = trange(n_episodes)
            # to record reward for each episode
            reward_array = np.zeros(n_episodes)
            for i in t:
                  # initial state
                  s = self.env.reset()
                  # initial action
                  a = self.epsilon_greedy(Q, epsilon, s)
                  done = False
                  while not done:
                        s_, reward, done, _ = self.env.step(a)
                        a_ = self.epsilon_greedy(Q, epsilon, s_)
                        # update Q table
                        Q[s, a] += alpha * (reward + (gamma * Q[s_, a_]) - Q[s, a])
                        # update processing bar
                        if done:
                              t.set_description('Episode {} Reward {}'.format(i + 1, reward))
                              t.refresh()
                              reward_array[i] = reward
                              break
                        s, a = s_, a_
            self.env.close()
            # show Q table
            print('Trained Q Table:')
            print(Q)
            # show average reward
            avg_reward = round(np.mean(reward_array), 4)
            print('Training Averaged reward per episode {}'.format(avg_reward))
            return Q
