# README
## Combinatorial Auction-based Collaborative Transportation
This is a python package to simulate combinatorial auction-based horizontal carrier
collaboration for request reassignment.
It has been used to produce research results for the following papers:

- Elting, Steffen, Jan Fabian Ehmke, and Margaretha Gansterer. “Collaborative Transportation for Attended Home Deliveries.” Networks, January 1, 2024. https://doi.org/10.1002/net.22216.
- Elting, Steffen, Jan Fabian Ehmke, and Margaretha Gansterer. “Preference Learning for Efficient Bundle Selection in Horizontal Transport Collaborations.” European Journal of Operational Research, 2024. https://doi-org/10.1016/j.ejor.2025.02.002
- tba

To run the optimization, run src/CACT/main.py. There are three configuration points:

- command line arguments (number of parallel threads, behavior on error, the group_id tag value to use for mlfow) as defined in utility_module/argparse_utils.py
- the filtering options for instance selection in main.py (these may be included as command line arguments in the future)
- the hyerparameters for solving the problems as defined in solver_module.config.py. Beware that this is a large and complex configuration file that generates all necessary parameter settings (and their corresponding classes) before the solving starts. 

The instances' file paths and the hyperparameter confirguration are then passed to the workflow_module.execute_jobs() function that handles multithreading if applied.

The main classes of this library are:
- CAHDInstance: the class that holds all the VRP instance data (CAHD is a legacy name for Collaborative Attended Home Delivery)
- Solution: the class that holds the solution details, mainly it stores a list of Carrier elements that each holds a list of Tour elements themselves
- Solver (e.g. CollaborativeSolver) handles the high-level logical flow of the optimization run
- Auction handles the auction process (order release, bundle selection, bidding, winnder determination)