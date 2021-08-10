import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RandomSearch
from worker_hp_TEPS import MyWorker



parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=1000)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=7000)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=500)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
parser.add_argument('--worker', help='Flag to turn this into a worker process', default=True,  action='store_true')
args=parser.parse_args()


# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
if True:
    w = MyWorker(sleep_interval = 0.5, nameserver='127.0.0.1',run_id='example1')
    w.run(background=False)
    exit(0)


# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
import hpbandster.core.result as hpres
result_logger = hpres.json_result_logger(directory="logs", overwrite=True)

bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'example1', 
              min_budget=args.min_budget, max_budget=args.max_budget,
              result_logger = result_logger

           )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers = 1)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
Random.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
all_runs = res.get_all_runs()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))
