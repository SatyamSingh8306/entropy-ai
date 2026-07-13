"""Sample agent for CLI demos."""
import random


def agent(inp):
    r = random.random()
    if r < 0.8:
        return "A"
    if r < 0.95:
        return "B"
    return "C"
