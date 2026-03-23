#!/usr/bin/env python

# std libs

###############################################################################
# function class
###############################################################################
class Verbose():
    def __init__(self, level):
        self.level = level

    def __call__(self, level, s):
        if level<=self.level:
            print(s)



