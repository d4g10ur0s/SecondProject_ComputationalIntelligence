import random
import array

import os

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utility.dataReader import data_reader
from utility.dataReader import datasetToFolds

global dataShape#shape of data
global stances#means of stances
global myConstant#c constant
global vectorData#the data
global experimentPopulation

def arrayComp(arr1,arrs):
    for i in arrs :
        k=True
        for j in range(len(arr1)):
            if not arr1[j]==i[j]:
                k=False
        if not k :
            return False
    return True

def myCosineSimilarity(v , t):
    dot = sum(a*b for a,b in zip(v,t) )
    norm_arr1 = sum(a*a for a in v) ** 0.5
    norm_arr2 = sum(b*b for b in t) ** 0.5
    return dot/(norm_arr1*norm_arr2)

def evaluate(individual):
    global stances
    global myConstant
    myConstant = 0.2
    # Do some hard computing on the individual
    a = myCosineSimilarity(individual[0][0],stances[1].tolist())
    b=0
    for i in stances:
        if i==1:
            pass
        else:
            b += myCosineSimilarity(individual[0][0],stances[i])
    return (( a + myConstant * (1 - 1/4 * b) )/( 1 + myConstant ),)


def getRandomVector():
    global dataShape
    global vectorData

    return vectorData.iloc[np.random.randint(low=dataShape , size=1)][:].values

def main():
    #1. preprocess data
    global dataShape
    dataShape = 1000
    global vectorData
    global experimentPopulation
    data=None
    if os.path.exists(os.getcwd() + "\\utility\\processedDataset.csv"):
        data = pd.read_csv(os.getcwd()+"\\utility\\processedDataset.csv", delimiter=";",low_memory = False)
    else:
        data = data_reader()
    global stances
    stances = {}
    for i in range(1,6):
        stances[i] = data.iloc[:][data["class"]==i].drop(columns=["Unnamed: 0","class"],inplace=False).mean()


    candidateVectors = []
    theMeans = []
    breakMeans = []
    peiramata = [
    [20,0.6,0],
    [20,0.6,0.01],
    [20,0.6,0.1],
    [20,0.9,0.01],
    [20,0.1,0.01],
    [200,0.6,0],
    [200,0.6,0.01],
    [200,0.6,0.1],
    [200,0.9,0.01],
    [200,0.1,0.01],
    ]
    #2. Initialize Genetic Algorithm
    #2.1 Define fitness function
    vectorData=data.drop(columns=["Unnamed: 0","class"])
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual",np.ndarray , fitness=creator.FitnessMax)
    #2.2 initializing the toolbox
    toolbox = base.Toolbox()
    #2.3 initialize individual with a random vector
    toolbox.register("vector", getRandomVector)
    toolbox.register("individual", tools.initRepeat,
                                   creator.Individual,
                                   toolbox.vector,
                                   n=1)
    toolbox.register("evaluate", evaluate)
    #2.4 create population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #2.5 selection method
    toolbox.register("select", tools.selRoulette)
    #2.6 crossover method
    toolbox.register("crover", tools.cxUniform, indpb=0.25)
    #2.7 mutation method , best values dont change
    toolbox.register("elitisism", tools.selBest)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
    for peirama,pnum in zip(peiramata,range(1,11)):# gia na sumplhrwsw ka8e grammh pinaka
        ##################################################
        theTops = []# ta kalutera fitness values se 10 fores
        theBrakes = []# genea termatismou
        crossoverChance = peirama[1]
        mutationChance = peirama[2]
        print("Peirama gia : ")
        print("Crossover chance : " + str(crossoverChance))
        print("Mutation chance : " + str(mutationChance))
        for mesos_oros_se_deka_peiramata in range(10):
            #3. Run Algorithm
            # evaluate all individuals in the population
            population = toolbox.population(n=peirama[0])
            fitnesses = list(toolbox.map(toolbox.evaluate, population))
            # assign the evaluated fitnesses to the population
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            '''
            The Genetic Script
            '''
            #3.4.1
            lastFitness = []
            newFitness = 0
            #3.4.2
            lastBestVectors = []
            broken = 1000
            for g in range(1000):
                #3.1 Select the next generation individuals
                parents = toolbox.select(population,int(len(population)/2))
                elit = [ind[0][0] for ind in toolbox.elitisism(population, int(len(population)/2))]#select the elit
                offspring = [toolbox.clone(ind) for ind in parents]
                #3.2 perform crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.uniform(0, 1) < crossoverChance:
                        toolbox.crover(child1[0][0], child2[0][0])
                        child1.fitness.values = evaluate(child1)
                        child2.fitness.values = evaluate(child2)
                #3.3 perform mutation
                for mutant in offspring:
                    if random.uniform(0, 1) < mutationChance and not arrayComp(mutant[0][0],elit):# mutation must not be in elit
                        mutant[0] = toolbox.mutate(mutant[0][0])
                        mutant.fitness.values = evaluate(mutant)

                #next gen is best pop
                population[int(len(population)/2):] = offspring
                #3.4 sun8hkes termatismou
                #3.4.1 ean to fitness den veltiw8hke
                #3.4.1 ean o kaluteros den veltiw8hke tis 10 teleutaies fores 10/1000 --> 0.01
                if g < 10:
                    lastFitness.append(evaluate(toolbox.clone(tools.selBest(population, k=1)[0]))[0])
                    lastBest = tools.selBest(population, k=1)[0]
                    lastBestVectors.append(lastBest[0][0])
                else:
                    newFitness = evaluate(toolbox.clone(tools.selBest(population, k=1)[0]))[0]
                    newBest = tools.selBest(population, k=1)[0]
                    countMe = 0
                    countMe2 = 0
                    for vec,lfitness in zip(lastBestVectors,lastFitness[len(lastFitness)-10:]):
                        if myCosineSimilarity(newBest[0][0],vec)==0:
                            countMe+=1
                        if abs(newFitness-lfitness) < 1e-5:
                            countMe2+=1
                    if countMe==10 or countMe2==10:
                        #broken.append(g)
                        broken=g
                        break
                    else:
                        #this list is like a queue
                        lastBestVectors.pop(0)
                        lastBestVectors.append(newBest[0][0])
                        #lastFitness.pop(0)
                        lastFitness.append(newFitness)
            # stop
            #create graph fitness / generation
            # create and store graph
            plt.plot(range(len(lastFitness)),lastFitness,label=mesos_oros_se_deka_peiramata)
            #plt.plot(theTops,theBrakes)
            #4. return best individual
            top = tools.selBest(population, k=1)[0]
            candidateVectors.append(top[0][0])
            #print("The best individual is: " + str(top[0][0]))
            print("Lasted at : " + str(broken) )
            theTops.append(evaluate(toolbox.clone(top))[0])# best fitness for mean value
            theBrakes.append(broken)# termatismos algori8mou
        # stop
        plt.title(str(peirama))
        print("To save : " + str(pnum))
        #plt.show()
        plt.legend(loc='upper right')
        plt.savefig("F:\\5oEtos\\EarinoEksamhno\\YpologistikhNohmosunh\\Project_B\\Graphs\\"+"Fitness_per_Generation"+str(pnum)+".png")
        plt.clf()
        theMeans.append(pd.DataFrame(data=theTops).mean())
        breakMeans.append(pd.DataFrame(data=theBrakes).mean())
        # create and store graph
        plt.plot(range(10),pd.DataFrame(data=theTops)/pd.DataFrame(data=theBrakes))
        #plt.plot(theTops,theBrakes)
        plt.title(str(peirama))
        print("To save : " + str(pnum))
        #plt.show()
        plt.savefig("F:\\5oEtos\\EarinoEksamhno\\YpologistikhNohmosunh\\Project_B\\Graphs\\"+"Experiment_Plot"+str(pnum)+".png")
        plt.clf()
    print("Fitness Mean : " + str(theMeans))
    print("Algorithm End : " + str(breakMeans))
    # create and store graph
    #plt.plot(breakMeans,theMeans)
    #plt.plot(theTops,theBrakes)
    #plt.title("The Means")
    #print("To save : " + str(pnum))
    #plt.show()
    #plt.savefig("F:\\5oEtos\\EarinoEksamhno\\YpologistikhNohmosunh\\Project_B\\Graphs\\theMeans.png")
    #plt.clf()
    pd.DataFrame(data=candidateVectors).to_csv(path_or_buf="F:\\5oEtos\\EarinoEksamhno\\YpologistikhNohmosunh\\Project_B\\generatedVectors.csv", sep=';')


if __name__ == "__main__":
    main()
