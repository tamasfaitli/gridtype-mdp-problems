import abc
import numpy as np

class MDP(metaclass=abc.ABCMeta):
    def __init__(self):
        self.states, self.map   = self.__states()
        self.actions            = self.__actions()
        self.n_states           = len(self.states)
        self.n_actions          = len(self.actions)
        self.transition_prob    = self.__transition_probabilities()
        self.rewards            = self.__rewards()

    @abc.abstractmethod
    def __states(self):
        raise NotImplementedError("States are not defined!")

    @abc.abstractmethod
    def __actions(self):
        raise NotImplementedError("Actions are not defined!")

    @abc.abstractmethod
    def __transition_probabilities(self):
        raise NotImplementedError("Transition probabilities are not defined!")

    @abc.abstractmethod
    def __rewards(self):
        raise NotImplementedError("Rewards are not defined!")

    @abc.abstractmethod
    def __move(self, state, action, aux=None):
        raise NotImplementedError("State transition is not defined!")

    @abc.abstractmethod
    def __end_condition(self, s, next_s):
        ''' Method to evaluate whether some end conditions met during
            the simulation

        :param s:       Current state
        :param next_s:  Next state
        :return: flag:  A flag
        '''
        raise NotImplementedError("End condition for simulation is not defined!")

    @abc.abstractmethod
    def __simulate_condition(self, flag, limit=None):
        ''' Working together with end condition. The two together can
            implement various ways to finish simulating the MPD.
            e.g. maze problem, the agent dies or exits the maze

        '''
        raise NotImplementedError("Simulate condition has not been implemented!")

    def simulate(self, start, policy, limit=None):
        path = []
        flag = 0
        # Initialize current state and time
        time = 0
        s = self.map[start]
        # Add the starting position in the maze to the path
        path.append(start)
        while self.__simulate_condition(flag, limit):
            # Move to next state given the policy and the current state
            # time-varying policy
            if len(policy.shape) == 2:
                if time >= policy.shape[1]:
                    break
                next_s = self.__move(s, policy[s, time])
            # time invariant policy
            else:
                next_s = self.__move(s, policy[s])

            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])

            flag = self.__end_condition(s, next_s)

            # Update time and state for next iteration
            time += 1
            s = next_s

        return path, flag

    # def animate(self, renderer, path, policy=None, V=None):
    #     raise NotImplementedError("Animation has not been implemented!")

    def solve_dynamic_programming(self, T):
        ''' Solves the MDP using dynamic programming

        :param T: horizon length
        :return: V: Optimal time-varying values at every state
        :return: policy - Optimal time-varying policy at every state
        '''
        V       = np.zeros((self.n_states, T+1))
        policy  = np.zeros((self.n_states, T+1))


        Q               = np.copy(self.rewards)
        V[:, T]         = np.max(Q,1)
        policy[:, T]    = np.argmax(Q,1)

        for t in range(T-1, -1, -1):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    Q[s, a] = self.rewards[s,a] + np.dot(self.transition_prob[:,s,a],V[:,t+1])
            V[:,t] = np.max(Q,1)
            policy[:,t] = np.argmax(Q,1)

        return V, policy

    def solve_value_iteration(self, gamma, epsilon):
        ''' Solve MPD using value iteration

        :param gamma:       Discount factor
        :param epsilon:     Accuracy of the value iteration procedure
        :return: V:         Optimal value for every state
        :return: policy:    Optimal policy for every state
        '''
        V = np.zeros(self.n_states)
        Q = np.zeros((self.n_states, self.n_actions))

        n = 0
        tol = (1-gamma)*epsilon/gamma

        # Initialization of the VI
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Q[s, a] = self.rewards[s, a] + \
                          gamma * np.dot(self.transition_prob[:, s, a], V)
        BV = np.max(Q, 1)

        # Iterate until convergence
        while np.linalg.norm(V - BV) >= tol and n < 200:
            # Increment by one the numbers of iteration
            n += 1
            # Update the value function
            V = np.copy(BV)
            # Compute the new BV
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    Q[s, a] = self.rewards[s, a] \
                            + gamma * np.dot(self.transition_prob[:, s, a], V)
            BV = np.max(Q, 1)
            # Show error
            # print(np.linalg.norm(V - BV))

        # Compute policy
        policy = np.argmax(Q, 1)
        # Return the obtained policy
        return V, policy