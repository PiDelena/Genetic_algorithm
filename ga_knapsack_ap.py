"""
Genetic Algorithm

@author: Paola Itzel Delena García
"""
import time
import numpy as np
import matplotlib.pyplot as plt

class Genetic_algorithm():	
	
    def __init__(self, weights, profits, opt, C, p, pm, pc, tgen):	
        self.C = C # Knapsack capacity
        self.weights = np.array(weights) # Weights
        self.profits = np.array(profits) # Profits
        self.opt = opt # Optimum
        self.parents = [] # Parents
        self.ev = [] # Population Evaluated
        self.best_p = [] # Best Parents
        self.generation = 1 # Generations
        self.p = p # Population Size
        self.nobjects = len(weights) 
        self.best_solution = [] # Best Solution
        self.pm = pm # Mutation Probability
        self.pc = pc # Crossover Probability
        self.convergence = [] # Convergence
        self.tgen = tgen # Total generations
        self.initialization()
        
    def initialization(self):
        print('------------ GENETIC ALGORITHM ------------')
        self.parents = np.random.randint(0,2,size=(self.p, self.nobjects))    
    
    def fitness(self, item):
        sum_w = np.sum(item*self.weights)
        sum_p = np.sum(item*self.profits)
        if (sum_w > self.C):
            return -1
        else: 
            return sum_p
        
    def evaluation(self, population):
        ev_pop = np.zeros((self.p, self.nobjects + 1))
        for e in range(len(population)):
            ft = self.fitness(population[e])
            ev_pop[e][0] = ft
            ev_pop[e][1:] = population[e]        
        return ev_pop, ev_pop[:,1:]
        
    def crossover(self, ch1, ch2):
        r = np.random.uniform(0, 1)
        if r < self.pc:
            csite = np.random.randint(1, self.nobjects) # Crossover site
            ch1 = np.concatenate((ch1[:csite], ch2[csite:]))
            ch1 = np.concatenate((ch2[:csite], ch1[csite:]))
        return ch1, ch2

    def mutation(self, ch):
        for i in range(self.nobjects):		
            r = np.random.uniform(0, 1)
            if r < self.pm:
                if ch[i] == 1:
                    ch[i] = 0
                else: 
                    ch[i] = 1
        return ch
    
    def store_best_solution(self):
        best_now = self.ev[self.ev[:,0].argmax()]
        if(best_now[0] > self.best_solution[0]):
            self.best_solution = best_now.copy()
    
    def selection(self, n_parents, method='roulette'): 
        if method == 'roulette':
            # Roulette Selection
            fit = np.where(self.ev[:,0] < 0, 0, self.ev[:,0])
            if np.sum(fit) > 0:
                prob = fit/np.sum(fit)
                sel_parents = np.random.choice(range(self.p), n_parents, p=prob)
            else:
                sel_parents = np.random.choice(range(self.p), n_parents, replace=False)
            
        elif method == 'random':
            # Random Selection
            sel_parents = np.random.choice(range(self.p), n_parents, replace=False)
        return sel_parents
        
    def main(self):
        # Evaluate the population
        self.ev , self.best_p = self.evaluation(self.parents)	

        # Store the best solution
        if (self.generation == 1):
            self.best_solution = self.ev[self.ev[:,0].argmax()]
        else:
            self.store_best_solution()
        
        print('\nBest solution:', self.best_solution[0],'-', 
              (self.best_solution[1:]).astype(int))
        self.convergence.append(self.best_solution[0])
        
        # Selection
        sel_p = self.p//2 # Number of required parents
        flag = 0
        if (self.p%2 == 1):
            sel_p += 1
            flag = 1    
        sel_par = self.selection(sel_p, method='roulette')
        new_pop = []
        for i in range(0, sel_p):
            if i < sel_p-1:
                p1 = self.best_p[sel_par[i]].copy()
                p2 = self.best_p[sel_par[i+1]].copy()    
            else:
                p1 = self.best_p[sel_par[i]].copy()
                p2 = self.best_p[0].copy()
                
            # Crossover
            child1, child2 = self.crossover(p1, p2)
            new_pop.append(child1)
            new_pop.append(child2)
            
        new_pop = np.array(new_pop)
        if flag == 1: # If population it's an odd number
            new_pop = new_pop[0:-1].copy()
        
        # Mutation
        for i in range(len(new_pop)):
            new_pop[i] = self.mutation(new_pop[i])

        # Stop Condition
        # if  self.generation == tgen or (self.opt == self.best_solution[1:]).all():
        if  self.generation == tgen:
            print("Total Generations:", self.generation)
            print("Total Weight:", np.sum((self.best_solution[1:])*self.weights))
            print("Total Profit:", self.best_solution[0], '\n')
            
        else:
            print("Generations:", self.generation)
            self.parents = new_pop.copy()
            self.generation += 1
            self.ev = []
            self.best_p = []
            self.main()
    
    def plot_convergence(self):
        plt.figure(0)
        plt.scatter(np.arange(1,len(self.convergence)+1), self.convergence, color='crimson')
        plt.plot(np.arange(1,len(self.convergence)+1), self.convergence, color='crimson')
        plt.xlabel('Generations')
        plt.ylabel('Weight')
        plt.title('Convergence')
        plt.show()
        
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
        
    elif items == 0:
        C = 350
        # Huevos, Pan, Leche, Tortilla, Atun, Manzana, Aguacate
        weights = [149, 72.9, 152, 95.9, 98.8, 61, 160]
        profits = [76, 40, 79, 53, 91, 31, 69]
        optimum = [0, 0, 1, 1, 1, 0, 0]
        
    return C, weights, profits, optimum
    
def info(gens, t, hist='False'):
    # Histogram
    if hist == 'True':
        plt.figure('Histogram')
        plt.hist(x=gens, range=(0,200), bins=20, color='crimson', rwidth=0.85)
        plt.title('Generations Histrogram')
        plt.xlabel('Generations')
        plt.ylabel('Frequency')
        plt.show()
    
    # Time and Generations    
    avg_g = np.sum(gens)/len(gens)
    print('Average Generations:', avg_g)
    avg_t = np.sum(t)/len(t)
    print('Average Time:', avg_t, 's')
    return avg_g, avg_t
        
# Knapsack Problem
C, weights, profits, opt = knapsack_var(10)

# ------------ Genetic Algorithm ------------
p = 40 # Population size
pm = 0.1 # Mutation Probability
pc = 1 # Crossover Probability
tgen = 20 # Total Generations

gens = []; t_run = []
for _ in range(1):
    start = time.time()
    ga = Genetic_algorithm(weights, profits, opt, C, p, pm, pc, tgen)
    ga.main()
    end = time.time()
    t_run.append(end-start)
    gens.append(ga.generation)

info(gens, t_run, hist='False') # Info and histogram
ga.plot_convergence() # Convergence

