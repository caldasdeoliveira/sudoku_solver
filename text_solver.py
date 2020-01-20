
import pandas as pd
import numpy as np

from random import randrange
import constant

def load_dataset(path_to_csv):
    ''' load dataset into numpy array. This project uses the structure from https://www.kaggle.com/bryanpark/sudoku
        input:
            system path to a csv file following the structure of https://www.kaggle.com/bryanpark/sudoku
        output:
            tuple with 2 numpy.arrays with all puzzles and all solutions
    '''
    quizzes = np.zeros((1000000, 81), np.int32)
    solutions = np.zeros((1000000, 81), np.int32)
    for i, line in enumerate(open(path_to_csv, 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
    quizzes = quizzes.reshape((-1, 9, 9))
    solutions = solutions.reshape((-1, 9, 9))
    return quizzes, solutions

def select_rand_puzzle(data):
    ''' selects a random puzzle
        input:
            tuple with 2 numpy.arrays with all puzzles and all solutions
        output:
            tuple with 2 numpy.arrays corresponding to a random puzzle and its solution
    '''
    idx = randrange(data[constant.QUIZ].shape[0])
    return (data[constant.QUIZ][idx], data[constant.SOLUTION][idx])

def puzzle_solution_check(puzzle_curr_state):
    ''' checks if a solved puzzle is correct
        input:
            numpy.array with dimension 9x9
        output:
            boolean that indicates if the solutions is correct or not
    '''
    assert (puzzle_curr_state<10).all(), "found value above 9"
    assert (puzzle_curr_state>0).all(), "found zero or negative value"
    for aux in range(0,9):
        _,counts_row = np.unique(puzzle_curr_state[aux,:],return_counts=True)
        _,counts_column = np.unique(puzzle_curr_state[:,aux],return_counts=True)
        _,square = np.unique(puzzle_curr_state[aux//3*3:aux//3*3+2,aux%3*3:aux%3*3+2],return_counts=True)
        if (counts_row>1).any() or (counts_column>1).any() or (square>1).any():
            return False
    return True

def puzzle_finished_check(puzzle_curr_state):
    ''' checks if a puzzle is finished
        input:
            numpy.array with dimension 9x9
        output:
            boolean that indicates if the puzzle is finished or not
    '''
    if 0 in puzzle_curr_state:
        return False
    else:
        return puzzle_solution_check(puzzle_curr_state)

def solve_puzzle(puzzle_curr_state):
    ''' solves a sudoku puzzle
        input:
            numpy.array with dimension 9x9
        output:
            boolean that indicates if a solution exists or not
    '''
    if puzzle_finished_check(puzzle_curr_state):
        return True
    else:
        idx = (puzzle_curr_state==0).argmax()
        existing_values = set(puzzle_curr_state[:,idx%9])
        existing_values.update(set(puzzle_curr_state[idx//9,:]))
        existing_values.update(set(puzzle_curr_state[idx//9//3*3:idx//9//3*3+2,idx%9//3*3:idx%9//3*3+2].flatten()))
        possible_values = set(range(1,10))- existing_values
        if possible_values:
            for i in list(possible_values):
                #print(str(puzzle_curr_state[idx//9,idx%9]) + '->' + str(i) + ' at ' + str(idx))
                puzzle_curr_state[idx//9,idx%9] = i
                if solve_puzzle(puzzle_curr_state):
                    return True
                puzzle_curr_state[idx//9,idx%9] = 0
        else:
            return False
    