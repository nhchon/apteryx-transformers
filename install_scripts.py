'''
Runs install steps on first import.
'''

import os

from src.apteryx_transformers.GLOBALS import (MESSAGE)

def say_hi():
    print(MESSAGE)

if __name__ == '__main__':
    print('Running post-install scripts!')
    say_hi()
