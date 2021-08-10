#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import logging 
logging.basicConfig(level=logging.WARNING)
from nasbench import api
import nasbench
import tensorflow
import argparse
from algorithms import GA
from multiprocessing import Pool
from OOP_config import init_config
from configEncoding import encode
def dict_loop(dict_, list_):
    for ID, config in enumerate(list_):
        dict_[ID] = config
    return dict_
import csv
import os

def save2csv(filename : str, dictionary : dict, first = False,iter_num : int = 0):

    #If this is a new run or file doesnt exist, overwrite/create the file
    if os.path.exists(filename) and first == False:
        write_type = 'a'
    else:
        write_type = 'w' 

    
    with open(filename, write_type) as csvfile:
        writer = csv.writer(csvfile) 
        for i in dictionary:
            writer.writerow([iter_num,i,dictionary[i]])

configspace = init_config()
###Configuration Settings###
pop_size = 36
elite_size = 0.2
num_iter = 5
batch_size = 256

nasbench = api.NASBench('nasbench_data/nasbench_full.tfrecord')



population = configspace.sample_configuration(pop_size)
pop_dict = dict()
score_dict = dict()
pop_dict = dict_loop(pop_dict, population) 

config_file = "configs.csv"
score_file = "scores.csv"
first = True
for count,i in enumerate(range(num_iter)):
    pop = []
    while len(pop) < pop_size:
        config = configspace.sample_configuration(1) 
        cell = encode(config.get_dictionary())
        print("Model Spec: ",cell)
        print("Cell is valid: ", nasbench.is_valid(cell))
        if nasbench.is_valid(cell):
          pop.append(cell)
    results = []
    for i in pop: 
       results.append(nasbench.query(i))
    print(results)
                 
