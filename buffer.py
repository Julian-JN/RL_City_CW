from collections import namedtuple, deque
import numpy as np
import random
import torch


class ReplayMemory(object):
    def __init__(self, capacity, use_per=False, alpha=0.6, epsilon=0.001):
        self.use_per = use_per
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.epsilon = epsilon
        self.count = 0
        if self.use_per:
            self.alpha = alpha
            self.sum_tree = SumTree(capacity)
            self.max_priority = 1.0

    def push(self, transition):
        self.memory.append(transition)
        if self.use_per:
            self.sum_tree.add(self.max_priority, self.count)
            self.count = (self.count + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        is_weights = []
        if self.use_per:
            total_priority = self.sum_tree.total
            # print(f"Total Priority: {total_priority}")
            segment = total_priority / batch_size
            for i in range(batch_size):
                s = random.uniform(segment * i, segment * (i + 1))
                tree_idx, priority, idx = self.sum_tree.get(s)
                sampling_probability = priority / total_priority
                is_weight = (len(self.memory) * sampling_probability) ** -beta
                is_weights.append(is_weight)
                # print(idx)
                batch.append(self.memory[idx])
                idxs.append(tree_idx)
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
                adjusted_priority = (priority + self.epsilon) ** self.alpha
                self.max_priority = max(self.max_priority, adjusted_priority)
                self.sum_tree.update(idx, adjusted_priority)
        else:
            raise ValueError("Not using PER")

    def __len__(self):
        return len(self.memory)

# class SumTree:
#     """This will be binary tree stored as a list (self.tree), where:
#      - the experiences priorities are the leaves, stored in the second half of the list
#      - the remaining positions (first half) are the binary sums of children nodes
#      - the root tree (the first element) is the sum of all the elements"""
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = [0] * (2 * capacity - 1)
#         self.data = [None] * capacity
#         self.write = 0
#
#     def update(self, idx, priority):
#         """Updates the whole tree after receiving a new priority
#         for an specific tree index, by propagating the change."""
#         print(idx)
#         tree_idx = idx + self.capacity - 1
#         print(tree_idx)
#         change = priority - self.tree[tree_idx]
#         self.tree[tree_idx] = priority
#         tree_idx = (tree_idx - 1) // 2
#         while tree_idx >= 0:
#             self.tree[tree_idx] += change
#             tree_idx = (tree_idx - 1) // 2
#
#     def add(self, priority, data):
#         """Add a new priority to the tree, and update the tree."""
#         self.data[self.write] = data
#         self.update(self.write, priority)
#         self.write = (self.write + 1) % self.capacity
#
#     def get(self, s):
#         assert s <= self.total(), "Priority error"
#
#         tree_idx = 0
#         while True:
#             left = 2 * tree_idx + 1
#             right = left + 1
#             if left >= len(self.tree):  # >= would provoke a very hard to debug error hehe
#                 break
#             if s <= self.tree[left]:
#                 tree_idx = left
#             else:
#                 s -= self.tree[left]
#                 tree_idx = right
#         idx = tree_idx - self.capacity + 1
#         if s < 0 or s > self.total():
#             raise ValueError(f"Cumulative probability {s} is out of valid range")
#         if idx < 0 or idx >= self.capacity:
#             raise ValueError(f"Data index {idx} out of range. Tree index: {tree_idx}")
#         if tree_idx < 0 or tree_idx >= len(self.tree):
#             raise ValueError(f"Tree index {tree_idx} out of range.")
#         return tree_idx, self.tree[tree_idx], self.data[idx]  # Return the index of the data in the ReplayMemory
#
#     def total(self):
#         return self.tree[0]

# https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]
        self.nodes[idx] = value
        parent = idx
        while parent != 0:
            parent = (parent - 1) // 2
            self.nodes[parent] += change

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2
            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx], self.data[data_idx]