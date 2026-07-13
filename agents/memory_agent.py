class MemoryAgent:

    def __init__(self):

        self.memory = {}

    def save(self, key, value):

        self.memory[key] = value

    def load(self, key):

        return self.memory.get(key)

    def show(self):

        return self.memory

    def clear(self):

        self.memory.clear()