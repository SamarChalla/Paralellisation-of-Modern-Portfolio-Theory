# A real-value encoded GA using uniform crossover and boundary mutation
# Selection is based on elitism and roulette wheel 

import time
import numpy as np
import pandas as pd
import random
from pandas_datareader import data # pandas datareader is a sub-package that allows one to create a dataframe from various internet resources
import matplotlib.pyplot as plt


def get_data(tickers):
    portfolio = pd.DataFrame()
    
    for t in tickers:
        portfolio[t] = data.DataReader(t, data_source = 'yahoo', start='2019-02-01')['Adj Close']
        
    portfolio.columns = tickers
    returns = portfolio.pct_change()# np.log(portfolio / portfolio.shift(1))
    
    port_return = np.array(returns.mean() * 252)
    port_risk = returns.cov()
    
    return portfolio, port_return, port_risk



def generate_weights(inputs, population):
    n_assets = len(inputs.columns) # input is the portfolio
    array = np.empty((population, (n_assets + 2))) # population = no.of sets of weights that we choose to iterate everytime
    weights = [] # Initializing weights
    
    for i in range(0, population):
        weighting = np.random.random(n_assets) # array of size = no.of assets
        weighting /= np.sum(weighting) # making their total sum to 1(constraint)
        weights.append(weighting)
    weights = np.array(weights)
    
    for i in range(0, n_assets):
       array[:, i] = weights[:, i] # forming them into a data frame
       
    return array, n_assets



def fitness_func(weights, port_return, port_risk, n_assets, riskFree):
    fitness = []
    
    for i in range(0, len(weights)):
        w_return = (weights[i, 0:n_assets] * port_return) 
        w_risk = np.sqrt(np.dot(weights[i, 0:n_assets].T, np.dot(port_risk, weights[i, 0:n_assets]))) * np.sqrt(252)
        score = ((np.sum(w_return) * 100) - riskFree) / (np.sum(w_risk) * 100) # Sharpe ratio
        fitness.append(score)
        
    fitness = np.array(fitness).reshape(len(weights)) #fitness score is the sharpe ratio
    weights[:, n_assets] = fitness # now weights has score also in the n_assets column
    
    return weights


#divide the results into elite and non-elite results
def elitism(elitism_rate, fit_func_res, n_assets):
    sorted_ff = fit_func_res[fit_func_res[:, n_assets].argsort()]
    #print(fit_func_res[:,n_assets])
    elite_w = int(len(sorted_ff) * elitism_rate) # to calculate how many elite results will be chosen as parents
    elite_results = sorted_ff[-elite_w:] #elite results obtained from sorting total results
    non_elite_results = sorted_ff[:-elite_w] #non-elite results
    
    return elite_results, non_elite_results


#selecting from the non-elite results
def selection(parents, n_assets):     
    sol_len = int(len(parents) / 2)
    if (sol_len % 2) != 0: sol_len = sol_len + 1
    crossover_gen = np.empty((0, (n_assets + 2)))  #initializing the crossover generation
    
    for i in range(0, sol_len):
        parents[:, (n_assets + 1)] = np.cumsum(parents[:, n_assets]).reshape(len(parents))
        rand = random.randint(0, int(sum(parents[:, n_assets])))
        
        for i in range(0, len(parents)): nearest_val = min(parents[i:, (n_assets + 1)], key = lambda x: abs(x - rand))
        val = np.where(parents == nearest_val)
        index = val[0][0]
        
        next_gen = parents[index].reshape(1, (n_assets + 2))
        
        crossover_gen = np.append(crossover_gen, next_gen, axis = 0) 
        parents = np.delete(parents, (val[0]), 0)
        
    non_crossover_gen = crossover_gen.copy()
    
    return crossover_gen, non_crossover_gen



def crossover(probability, weights, assets):   
    for i in range(0, int((len(weights))/2), 2): 
        gen1, gen2 = weights[i], weights[i+1]
        gen1, gen2 = uni_co(gen1, gen2, assets, probability)
        weights[i], weights[i+1] = gen1, gen2
        
    weights = normalise(weights, assets)
    
    return weights
    


def uni_co(gen1, gen2, assets, crossover_rate):
    prob = np.random.normal(1, 1, assets)
    
    for i in range(0, len(prob)):
        if prob[i] > crossover_rate:
            gen1[i], gen2[i] = gen2[i], gen1[i]  
            
    return gen1, gen2



def mutation(probability, generation, assets): 
    weight_n = len(generation) * ((np.shape(generation)[1]) - 2)
    mutate_gens = int(weight_n * probability)
    
    if (mutate_gens >= 1):
        for i in range(0, mutate_gens):
            rand_pos_x, rand_pos_y = random.randint(0, (len(generation) - 1)), random.randint(0, (assets - 1))
            mu_gen = generation[rand_pos_x][rand_pos_y]
            mutated_ind = mu_gen * np.random.normal(0,1)
            generation[rand_pos_x][rand_pos_y] = abs(mutated_ind)
            generation = normalise(generation, assets)
        return generation
    else:
        return generation



def normalise(generation, assets):
    for i in range(0, len(generation)):
        generation[i][0:assets] /= np.sum(generation[i][0:assets])
    return generation



def next_gen(elites, children, no_cross_parents):
    weights = np.vstack((elites, children, no_cross_parents))
    return weights 



def optimal_solution(generations, assets):
    optimal_weights = generations[generations[:, (assets + 1)].argsort()]
  #  print(optimal_weights)
    return optimal_weights[0,0:-2]



def avg_gen_result(weights, n_assets):
    average = round(np.mean(weights[:, n_assets]), 2)
    maxi = round(np.max(weights[:,n_assets]),2)
    return average, maxi



def genetic_algorithm(tickers, risk_free_rate, population, generations, crossover_rate, mutation_rate, elite_rate):
    weights, port_return, port_risk = get_data(tickers)
    weights, n_assets = generate_weights(weights, population)
    

    for i in range(0, generations):
        results = fitness_func(weights, port_return, port_risk, n_assets, risk_free_rate)
        #print(results)
             
        elites, parents = elitism(elite_rate, results, n_assets)
        parents, no_cross_parents = selection(parents, n_assets)
        children = crossover(crossover_rate, parents, n_assets)
        children = mutation(mutation_rate, children, n_assets) 
        
        weights = next_gen(elites, children, no_cross_parents)
        
        avg_res, maxi = avg_gen_result(weights, n_assets)
        #print('Generation', i, ': Average Sharpe Ratio of', avg_res, 'from', len(weights), 'chromosomes')
     
        average_result.append(avg_res)
        max_result.append(maxi)
        
    opt_solution = optimal_solution(weights, n_assets)
    
    return opt_solution, average_result


# Function Inputs
start_time = time.time()
tickers = ['HDFCBANK', 'RELIANCE','INFY','TCS','ITC','AXISBANK']
tickers = [t +'.NS' for t in tickers]
population = 4
risk_free_rate = 2
generations = 1
crossover_rate = 0.4
mutation_rate = 0.01
elite_rate = 0.25
average_result = []
max_result =[]


# Run Function and Return Optimal Weights

optimal_weights,average_result = genetic_algorithm(tickers, risk_free_rate, population, generations, crossover_rate, mutation_rate, elite_rate)
#print(optimal_weights)
#print(np.sum(optimal_weights))
average_result = np.array(average_result)
max_result = np.array(max_result)
for i in range(len(tickers)):
	print(f'{tickers[i]}-{optimal_weights[i]}')

plt.plot(average_result)
plt.plot(max_result)
plt.xlabel('Generation')
plt.ylabel('Average sharpe ratio')
plt.show()

end_time = time.time()
print(f"Total time-{start_time-end_time}")
