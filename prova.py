# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:05:17 2024

@author: carlo
"""
import math
import numpy as np
from timebudget import timebudget
from multiprocessing import Pool

iterations_count = round(1e7)


def complex_operation(input_index):
    print("Complex operation. Input index: {:2d}\n".format(input_index))
    [math.exp(i) * math.sinh(i) for i in [1] * iterations_count]


@timebudget
def run_complex_operations(function, input, pool):
    """
    Function to compute squares of numbers in parallel.
    """
    # Map the square function to each number in parallel
    pool.map(function, input)

    # Close the pool to release resources
    pool.close()
    pool.join()


processes_count = 10

if __name__ == "__main__":
    processes_pool = Pool(processes_count)
    run_complex_operations(complex_operation, range(10), processes_pool)
