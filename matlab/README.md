# Illustration of code files

There are three folders in this folder - functions, dataset and results.
The codes here will produce experimental results with ADMM algorithm of specified dataset.

There are  code files:
1) 'run.m': this function is a main function, with reads data, implements ADMM algorithm and save results.
2) 'ADMM_algorithm.m': this script is called by 'run.m'. It implements all algorithms of ADMM.
3) 'read_uai.m': called by 'ADMM_algorithm.m'. Read the data in dataset uai files.
4) 'compute_log_potential.m': called by 'ADMM_algorithm.m'. Calculate objective values. 
5) 'compute_std_obj.m': called by 'ADMM_algorithm.m'. Calculate standard deviation of objective values.
6) 'project_shifted_Lp_ball.m': called by 'ADMM_algorithm.m'. Project extra variable node onto shifted Lp ball.
7) 'project_simplex.m': called by 'ADMM_algorithm.m'. Project a D-dimentional matrix into probability simplex.
8) 'generate_clock_str.m': called by 'run.m'. Record the time after finishing one test.

