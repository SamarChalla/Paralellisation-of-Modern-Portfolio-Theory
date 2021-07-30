from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np
import time
import pandas as pd
import random
from pandas_datareader import data # pandas datareader is a sub-package that allows one to create a dataframe from various internet resources
import matplotlib.pyplot as plt

def get_data(tickers):
    portfolio = pd.DataFrame()
    
    for t in tickers:
        portfolio[t] = data.DataReader(t, data_source = 'yahoo', start='2018-02-01')['Adj Close']
        
    portfolio.columns = tickers
    returns = portfolio.pct_change() # np.log(portfolio / portfolio.shift(1))
    
    port_return = np.array(returns.mean() * 252)
    port_risk = returns.cov()
    
    return portfolio, port_return, port_risk
    
def generate_weights(inputs, population):
    np.random.seed(rank)
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
    optimal_weights = generations[generations[:, (assets)].argsort()]
  #  print(optimal_weights)
    return optimal_weights[len(generations)-1,0:-1]



def avg_gen_result(weights, n_assets):
    average = round(np.mean(weights[:, n_assets]), 2)
    maxi = round(np.max(weights[:,n_assets]),2)
    return average, maxi

#inputs for the code
tickers = ['HDFCBANK', 'RELIANCE','INFY','TCS','ITC','AXISBANK']
tickers = [t +'.NS' for t in tickers]
population = 500
risk_free_rate = 2
generations = 400
crossover_rate = 0.4
mutation_rate = 0.01
elite_rate = 0.25
average_result = []
max_result =[]

weights_lcl = None
port_return = None
port_risk = None
weights = None
n_assets = 0
avg_res_glo = None

# if rank ==0:
#     #form population
#     print(weights)
#     #print(np.shape(weights))
#     #comm.scatter(weights,weights_lcl,root = 0)
    
# else:
#     weights = None
start_time_f = time.time()
start_time = time.time()    
weights, port_return, port_risk = get_data(tickers)
end_time = time.time()
#print("For Rank {} time taken for Getting Data = {}".format(rank,end_time-start_time))

start_time = end_time
weights, n_assets = generate_weights(weights,int(population/size))
end_time = time.time()
#print("For Rank {} time taken for Generating weights= {}".format(rank,end_time-start_time))

# print(weights)
# print(len(weights))
weights_lcl = weights
comm.bcast(n_assets , root = 0)    




#now use pop_local to get new_gen
#print("Entering into For LOOP")
start_time = time.time()    
for i in range(0, generations):
        results = fitness_func(weights_lcl, port_return, port_risk, n_assets, risk_free_rate)
             
        elites, parents = elitism(elite_rate, results, n_assets)
        parents, no_cross_parents = selection(parents, n_assets)
        children = crossover(crossover_rate, parents, n_assets)
        children = mutation(mutation_rate, children, n_assets) 
        
        weights_lcl = next_gen(elites, children, no_cross_parents)

        avg_res, maxi = avg_gen_result(weights_lcl, n_assets)
        average_result.append(avg_res)
        max_result.append(maxi)


        end_time = time.time()
        #print("For Rank {} time taken for Fitness Eval = {}".format(rank,end_time-start_time))

avg_res_glo = comm.gather(average_result,root = 0)
max_res_glo = comm.gather(max_result, root = 0)
weights_glo = comm.gather(weights_lcl, root = 0)
        
        #avg_res, maxi = avg_gen_result(weights, n_assets)
        #print('Generation', i, ': Average Sharpe Ratio of', avg_res, 'from', len(weights), 'chromosomes')
     
        #average_result.append(avg_res)
        #max_result.append(maxi)
        
if rank ==0:
	weights_glo = np.vstack(weights_glo) # vertical stacking to make it a 2d numpy array
	opt_solution = optimal_solution(weights_glo, n_assets)
	#print(weights_glo[:,n_assets])
	avg_res_glo = np.array(avg_res_glo)
	np.reshape (avg_res_glo, (1,generations*size))
	max_res_glo = np.array(max_res_glo)
	np.reshape (max_res_glo, (1,generations*size))
	#print(max_res_glo)
	
	Avg_sharpe_ratio = np.average(avg_res_glo)
	optimum_sharpe_ratio = np.amax(max_res_glo)
	print(f"Average sharpe ratio for all the generations - {Avg_sharpe_ratio}")
	print(f"Maximum sharpe ratio for all the generations-{optimum_sharpe_ratio}")
	for i in range(len(tickers)):
		print(f"{tickers[i]}-{opt_solution[i]}")	
			
	print(f'Optimum solution {opt_solution[0:6]} with sharpe ratio of {opt_solution[6]}')
	#print(np.sum(opt_solution[0:6]))
	end_time_f = time.time()
	print(f"Total time is {end_time_f - start_time_f}")
	#for i in range(size):
		#plt.plot(avg_res_glo[i])
		#plt.plot(max_res_glo[i])
	#plt.xlabel('Generation')
	#plt.ylabel('Sharpe ratio')
	#plt.show()
	    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
