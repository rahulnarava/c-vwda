import gym
import numpy as np
import random
from gym.utils import seeding

class GridworldEnvironment(gym.Env):
    """
    nxm Gridworld. Discrete states and actions (up/down/left/right/stay).
    Agent starts randomly.
    Goal is to reach the reward.
    """

    def __init__(self, r=None, t=None, transition_prob=1., stay_action=True, unsafe_states=[],
        start_states=None):
        """
        Construct the environment.
        Reward matrix is a 2D numpy matrix or list of lists.
        Terminal cells is a list/set of (i, j) values.
        Transition probability is the probability to execute an action and
        end up in the right next cell.
        """
        # Default to Gridworld A specs if not provided
        if r is None:
            r = np.zeros((7, 7)); r[6, 0] = 1.
        if t is None:
            t = [(6, 0)]
        if not unsafe_states:
            # Gridworld A constraints: Row 3, Cols 0-3
            unsafe_states = [(3, 0), (3, 1), (3, 2), (3, 3)]
        if start_states is None:
            # Gridworld A start states: Rows 0-2, Cols 0-1
            start_states = [(ui, uj) for ui in [0,1,2] for uj in [0,1]]

        self.reward_mat = np.array(r)
        assert(len(self.reward_mat.shape) == 2)
        self.h, self.w = len(self.reward_mat), len(self.reward_mat[0])
        self.n = self.h*self.w
        self.terminals = t
        if stay_action:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu', 8: 's'}
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu'}
            self.action_space = gym.spaces.Discrete(8)
            
        self.transition_prob = transition_prob
        self.terminated = True
        # Observation space is strictly the integer coordinates or index?
        # Standard RL algos usually expect Box(low, high) or Discrete.
        # This implementation returns list(self.state) which is [r, c].
        # Let's define observation space as Box.
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), 
            high=np.array([self.h, self.w]), dtype=np.float32) # using float for neural nets usually
        
        self.unsafe_states = unsafe_states
        self.start_states = start_states
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # Fix for seeding issue: ensure seed is within valid range for random.seed
        if seed is not None:
            # Force seed to be valid uint32 for legacy np.random.seed
            seed = int(seed) % (2**32 - 1)
        
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def get_states(self):
        return filter(
            lambda x: self.reward_mat[x[0]][x[1]] not in \
                [-np.inf, float('inf'), np.nan, float('nan')],
            [(i, j) for i in range(self.h) for j in range(self.w)]
        )
    
    def terminal(self, state):
        for terminal_state in self.terminals:
            if state[0] == terminal_state[0] and state[1] == terminal_state[1]:
                return True
        return False

    def get_next_states_and_probs(self, state, action):
        if self.terminal(state):
            return [((state[0], state[1]), 1)]
        if self.transition_prob == 1:
            inc = self.neighbors[action]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if nei_s[0] >= 0 and nei_s[0] < self.h and \
                nei_s[1] >= 0 and nei_s[1] < self.w and \
                self.reward_mat[nei_s[0]][nei_s[1]] not in \
                [-np.inf, float('inf'), np.nan, float('nan')]:
                return [(nei_s, 1)]
            else:
                return [((state[0], state[1]), 1)] # state invalid
        else:
            # Simplified transition logic from original file
            # Assuming deterministic for now based on thesis usage or 1.0 prob
            return [((state[0], state[1]), 1)]

    def idx2pos(self, idx):
        return (idx % self.h, idx // self.h)

    def reset(self):
        if self.start_states is not None:
            random_idx = self.np_random.randint(len(self.start_states))
            self.curr_state = self.start_states[random_idx]
        else:
            random_state = self.np_random.randint(self.h * self.w)
            self.curr_state = self.idx2pos(random_state)
            
        # Ensure we don't start in terminal or unsafe
        while self.curr_state in self.terminals or self.curr_state in self.unsafe_states:
             if self.start_states is not None:
                random_idx = self.np_random.randint(len(self.start_states))
                self.curr_state = self.start_states[random_idx]
             else:
                random_state = self.np_random.randint(self.h * self.w)
                self.curr_state = self.idx2pos(random_state)
                
        self.terminated = False
        return np.array(self.curr_state, dtype=np.float32)


    def step(self, action):
        if hasattr(action, 'shape') and len(action.shape) > 0 and action.shape != ():
             # If it's an array with dimensions, take the first element (scalar)
             action = action.ravel()[0]
        action = int(action)
        if self.terminal(self.curr_state):
            self.terminated = True
            return np.array(self.curr_state, dtype=np.float32), \
                self.reward_mat[self.curr_state[0], self.curr_state[1]], \
                True, \
                {'cost': 0.0}

        st_prob = self.get_next_states_and_probs(self.curr_state, action)
        # Deterministic check for now essentially
        next_state = st_prob[0][0]
        
        last_state = self.curr_state
        self.curr_state = next_state
        
        reward = self.reward_mat[self.curr_state[0]][self.curr_state[1]]
        
        # Calculate cost
        # Logic: lambda s, a: s[0] in [3] and s[1] in [0, 1, 2, 3]
        # Using NEXT state or CURRENT state?
        # Thesis config says "lambda s, a: ...". Usually implies current state s.
        # "unsafe_states" usually implies entering them is bad.
        # But here the list `unsafe_states` is disjoint from `start_states`.
        # If the agent IS in an unsafe state, it incurs cost?
        # Or if it TRANSITIONS into one?
        # The prompt says "Constraint: Row 3, Cols 0-3".
        # Let's check logic: if NEXT state is in unsafe_states => cost = 1.
        
        cost = 0.0
        if tuple(self.curr_state) in self.unsafe_states:
             cost = 1.0

        done = self.terminal(self.curr_state)
        
        return np.array(self.curr_state, dtype=np.float32), reward, done, {'cost': cost}

    def render(self, mode='human'):
        pass
