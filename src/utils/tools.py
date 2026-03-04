import random

def get_random():
    random_number = random.uniform(0, 1)
    while random_number == 0 or random_number == 1:
        random_number = random.uniform(0, 1)
    return random_number