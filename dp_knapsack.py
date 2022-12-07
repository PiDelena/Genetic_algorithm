"""
Dynamic Programming: Subset Sums Algorithm
@author: Paola Itzel Delena García
"""
import numpy as np
# import time
# import matplotlib.pyplot as plt

class Subset_Sum():
    
    def __init__(self, weights, C, V):
        self.weights = weights # Weights
        self.wlen = len(weights)
        self.C = C # Knapsack capacity
        self.S = [[0 for _ in range(C+1)] for _ in range(self.wlen+1)] # Create subsets arrays
        self.V = V # Profits
        
    def main(self):
    
        # Build the table cell by cell
        for rI in range(1, self.wlen + 1):
            for cI in range(1, self.C + 1):
               if cI < self.weights[rI - 1]:
                   self.S[rI][cI]= self.S[rI - 1][cI]
               else:
                   # Include or exclude item
                   self.S[rI][cI]= max(self.S[rI - 1][cI], self.V[rI - 1] + self.S[rI - 1][cI - self.weights[rI - 1]])
    
    def find_solution(self):
        
        # Build the solution based on the table
        sol = np.zeros(len(self.weights), dtype=int)
        cI = self.C
        rI = self.wlen
        # Search in the table from end to beginning
        while cI >= 0 and rI >= 0:
            if self.S[rI][cI] == self.S[rI - 1][cI]: # If the value is equal to the value in the top row
                rI = rI - 1
            else: # If it's different, then that value is part of the solution
                sol[rI-1] = 1
                cI = cI - self.weights[rI-1]
                rI = rI - 1 
        tW = np.sum(sol*np.array(self.weights))
        tV = np.sum(sol*np.array(self.V))
        print('Best Solution:', sol)
        print('Total Weight:', tW)
        print('Total Profit:', tV)
        return tW, tV, sol

def knapsack_var(items):
    
    if items == 3:
        C = 5
        weights = [3, 4, 2]
        profits = [4, 5, 3]
        optimum = [1, 0, 1]
    
    if items == 4:
        C = 14
        weights = [4, 6, 5, 4]
        profits = [6, 11, 3, 5]
        optimum = [1, 1, 0, 1]
    
    if items == 5:
        C = 26
        weights = [12, 7, 11, 8, 9]
        profits = [24, 13, 23, 15, 16]
        optimum = [0, 1, 1, 1, 0]
        
    elif items == 6:
        C = 190
        weights = [56, 59, 80, 64, 75, 17]
        profits = [50, 50, 64, 46, 50, 5]
        optimum = [1, 1, 0, 0, 1, 0]
    
    elif items == 7:
        C = 50
        weights = [31, 10, 20, 19, 4, 3, 6]
        profits = [70, 20, 39, 37, 7, 5, 10]
        optimum = [1, 0, 0, 1, 0, 0, 0]
        
    elif items == 8:
        C = 80
        weights = [33, 12, 24, 41, 19, 16, 35, 7]
        profits = [25, 9, 15, 33, 17, 20, 22, 5]
        optimum = [1, 1, 0, 0, 1, 1, 0, 0]
    
    elif items == 9:
        C = 126
        weights = [24, 41, 29, 44, 53, 38, 63, 45, 28]
        profits = [15, 33, 49, 68, 60, 43, 52, 34, 16]
        optimum = [0, 0, 1, 1, 1, 0, 0, 0, 0]
        
    elif items == 10:
        C = 165
        weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
        profits = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
        optimum = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
        
    elif items == 11:
        C = 151
        weights = [31, 29, 44, 53, 38, 63, 87, 60, 43, 67, 31] 
        profits = [57, 39, 38, 60, 43, 67, 72, 54, 25, 33, 19]
        optimum = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    
    elif items == 12:
        C = 159
        weights = [31, 29, 44, 53, 38, 63, 65, 77, 60, 43, 67, 31] 
        profits = [57, 39, 38, 60, 43, 67, 84, 72, 54, 25, 33, 19]
        optimum = [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        
    elif items == 13:
        C = 187
        weights = [31, 29, 44, 53, 38, 52, 85, 89, 82, 60, 43, 67, 31] 
        profits = [57, 49, 68, 60, 43, 66, 84, 87, 72, 54, 25, 33, 19]
        optimum = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
        
    elif items == 14:
        C = 195
        weights = [23, 31, 29, 44, 53, 38, 63, 65, 35, 82, 60, 43, 67, 31] 
        profits = [12, 57, 49, 68, 60, 43, 67, 84, 40, 72, 54, 25, 33, 19]
        optimum = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        
    elif items == 15:
        C = 200
        weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82, 60, 43, 67, 31, 10] 
        profits = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72, 54, 25, 33, 19, 6]
        optimum = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        
    return C, weights, profits, optimum

# Knapsack Problem
C, weights, profits, opt = knapsack_var(10)

# Subset Sum
Ss = Subset_Sum(weights, C, profits)
Ss.main()
Ss.find_solution()
