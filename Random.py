#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import logging 
logging.basicConfig(level=logging.WARNING)
from nasbench import api
import nasbench
import tensorflow
import numpy as np
import argparse
from algorithms import GA
from multiprocessing import Pool
from OOP_config import init_config
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
pop_size = 1000
elite_size = 0.2
num_iter = 5
batch_size = 256

nasbench = api.NASBench('nasbench_data/nasbench_full.tfrecord')

def generate_cell(op_dict, input_dict, output_op_idx):
  cell_array = np.zeros((len(op_dict.keys())+2,len(op_dict.keys())+2))
  ops = [0]*len(op_dict.keys()) 
  ops.insert(0, "input")
  ops.append("output")
  # [op_idx, input_idx]
  for op_idx in op_dict:
      op = op_dict[op_idx]
      ops[op_idx] = op 
      for input_idx in input_dict[op_idx]:
        cell_array[op_idx,input_idx] = 1
  cell_array = np.triu(cell_array.T, 1)
  cell_array[output_op_idx,output_op_idx +1] = 1
  return cell_array, ops



def encode(hyperparameters):
  op_dict = {} 
  input_dict = {}
  for i in hyperparameters: 
    if "normal_cell" in i:
      if "type" in i:
        op_dict[int(i[-6])] = hyperparameters[i]
  for i in op_dict.keys():
    input_dict[i] = []
  for i in hyperparameters: 
    if "normal_cell" in i:
      if "input" in i:
        input_dict[int(i[-9])].append(hyperparameters[i])

  for i in input_dict:
    input_dict[i] = set(input_dict[i])
        
  output_op = max(op_dict.keys())
  
  cell_array, ops = generate_cell(op_dict, input_dict, output_op)
  # Query an Inception-like cell from the dataset.
  # cell = api.ModelSpec(
  #   matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
  #           [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
  #           [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
  #           [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
  #           [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
  #           [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
  #           [0, 0, 0, 0, 0, 0, 0]],   # output layer
  #   # Operations at the vertices of the module, matches order of matrix.
  # 
  # Querying multiple times may yield different results. Each cell is evaluated 3
  # times at each epoch budget and querying will sample one randomly.
  print(cell_array)
  cell = api.ModelSpec(cell_array, ops)
  return cell

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
        if nasbench.is_valid(cell):
          pop.append(cell)
    results = []
    for c,i in enumerate(pop):
      try:
         r = nasbench.query(i, epochs =4)
         print("Model", c, "Succeded")
         print("Original Spec: ", i.original_matrix, i.original_ops)
         print("Attempting to query: ",i.matrix, i.ops)
         print("Result: ", r)
         input()
         results.append(r)
      except:
         print("Model ",c, "failed")
    print(results)
                 
