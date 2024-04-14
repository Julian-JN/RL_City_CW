from collections import namedtuple, deque
import numpy as np
import random
import torch


class ReplayMemory(object):
    def __init__(self, capacity, use_per=False, alpha=0.6, epsilon=0.01):
        self.use_per = use_per
        self.memory = deque([], maxlen=capacity)
        self.epsilon = epsilon
        if self.use_per:
            self.alpha = alpha
            self.sum_tree = SumTree(capacity)
            self.max_priority = 1.0

    def push(self, transition):
        self.memory.append(transition)
        if self.use_per:
            self.sum_tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        is_weights = []
        if self.use_per:
            segment = self.sum_tree.total() / batch_size
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                idx, data = self.sum_tree.get(s)
                priority = self.sum_tree.tree[idx]
                sampling_probabilities = priority / self.sum_tree.total()
                is_weight = (len(self.memory) * sampling_probabilities) ** -beta
                is_weight /= max(is_weights) if is_weights else 1  # Normalize weights
                idxs.append(idx)
                batch.append(data)
                is_weights.append(is_weight)
        else:
            batch = random.sample(self.memory, batch_size)
            is_weights = [1.0] * batch_size  # Uniform weights
            idxs = None
        
        return batch, idxs, is_weights
        
    def update_priority(self, idxs, priorities):
        if self.use_per:
            for idx, priority in zip(idxs, priorities):
                self.max_priority = max(self.max_priority, priority) #max_priorities like this?
                self.sum_tree.update(idx, (priority + self.epsilon) ** self.alpha) #epsilon or not here?
        else:
            raise ValueError("Not using PER")

    def __len__(self):
        return len(self.memory)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            idx = left if s <= self.tree[left] else right
            s -= self.tree[left] if s > self.tree[left] else 0
        return idx, self.data[idx - self.capacity + 1]

    def total(self):
        return self.tree[0]