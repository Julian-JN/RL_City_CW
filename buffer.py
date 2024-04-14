from collections import namedtuple, deque
import numpy as np
import random
import torch


class ReplayMemory(object):
    def __init__(self, capacity, use_per=False, alpha=0.6, epsilon=0.01):
        self.use_per = use_per
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.epsilon = epsilon
        if self.use_per:
            self.alpha = alpha
            self.sum_tree = SumTree(capacity)
            self.max_priority = 1.0

    def push(self, transition):
        self.memory.append(transition)
        if self.use_per:
            self.sum_tree.add((self.max_priority+ self.epsilon) ** self.alpha)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        is_weights = []
        if self.use_per:
            total_priority = self.sum_tree.total()
            segment = total_priority / batch_size
            for i in range(batch_size):
                s = random.uniform(segment * i, segment * (i + 1))
                tree_idx, idx = self.sum_tree.get(s)
                priority = self.sum_tree.tree[tree_idx]
                sampling_probability = priority / total_priority
                is_weight = (len(self.memory) * sampling_probability) ** -beta
                is_weights.append(is_weight)
                batch.append(self.memory[idx])
                idxs.append(idx)
            max_weight = max(is_weights)
            is_weights = [w / max_weight for w in is_weights]
        else:
            batch = random.sample(self.memory, batch_size)
            is_weights = [1.0] * batch_size
            idxs = None
        return batch, idxs, is_weights
        
    def update_priority(self, idxs, priorities):
        if self.use_per:
            for idx, priority in zip(idxs, priorities):
                self.max_priority = max(self.max_priority, priority)
                adjusted_priority = (priority + self.epsilon) ** self.alpha
                if 0 <= idx < self.capacity:
                    self.sum_tree.update(idx, adjusted_priority)
                else:
                    print(f"Index {idx} out of range.")
                    raise IndexError("Priority index out of range.")
        else:
            raise ValueError("Not using PER")

    def __len__(self):
        return len(self.memory)

class SumTree:
    """This will be binary tree stored as a list (self.tree), where:
     - the experiences priorities are the leaves, stored in the second half of the list
     - the remaining positions (first half) are the binary sums of children nodes
     - the root tree (the first element) is the sum of all the elements"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.write = 0

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority):
        self.update(self.write, priority)
        self.write = (self.write + 1) % self.capacity

    def get(self, s):
        tree_idx = 0
        while True:
            left = 2 * tree_idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                tree_idx = left
            else:
                s -= self.tree[left]
                tree_idx = right
        idx = tree_idx - self.capacity + 1
        if idx < 0 or idx >= self.capacity:
            raise ValueError(f"Index {idx} out of range. Tree index: {tree_idx}")
        if tree_idx < 0 or tree_idx >= len(self.tree):
            raise ValueError(f"Tree index {tree_idx} out of range.")
        return tree_idx, idx  # Return the index of the data in the ReplayMemory 

    def total(self):
        return self.tree[0]