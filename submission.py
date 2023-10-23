import random

import numpy as np
import time

import Gobblet_Gobblers_Env as gge
from scipy.spatial import distance
not_on_board = np.array([-1, -1])


# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final is 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


def timeout(final_time):
    return final_time - time.time() < 0.0


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns


def is_winner(is_final, agent_id):
    if (is_final == "1" and agent_id == 0) or (is_final == "2" and agent_id == 1):
        return 12
    elif (is_final == "2" and agent_id == 0) or (is_final == "1" and agent_id == 1):
        return -12
    else:
        return None


def smart_heuristic(state, agent_id, time_limit, depth):
    sum_pawns = 0
    middle = [np.array([1, 1])]
    corners = [np.array([0, 0]), np.array([0, 2]), np.array([2, 0]), np.array([2, 2])]
    sides = [np.array([0, 1]), np.array([1, 0]), np.array([1, 2]), np.array([1, 1])]

    winner_value = is_winner(gge.is_final_state(state), agent_id)
    if winner_value:
        return winner_value

    if (time_limit - time.time() < 0.0) or (depth == 0):
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                # check if position is in the middle
                if np.any(np.all(value[0] == middle, axis=1)):
                    # 4 winning possibilities:
                    sum_pawns += 4
                # check if position is in the corners
                elif np.any(np.all(value[0] == corners, axis=1)):
                    # 3 winning possibilities:
                    sum_pawns += 3
                # check if position is in the sides
                elif np.any(np.all(value[0] == sides, axis=1)):
                    # 2 winning possibilities:
                    sum_pawns += 2

        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                # check if position is in the middle
                if np.any(np.all(value[0] == middle, axis=1)):
                    # 4 winning possibilities:
                    sum_pawns -= 4
                # check if position is in the corners
                elif np.any(np.all(value[0] == corners, axis=1)):
                    # 3 winning possibilities:
                    sum_pawns -= 3
                # check if position is in the sides
                elif np.any(np.all(value[0] == sides, axis=1)):
                    # 2 winning possibilities:
                    sum_pawns -= 2

        return sum_pawns * -1 if agent_id else sum_pawns
    return None


# def smart_heuristic(state, agent_id):
#     sum_pawns = 0
#     sum_neighbours = 0
#     if agent_id == 0:
#         # a list of all the pieces positions and pieces
#         player_positions_on_board = list(state.player1_pawns.items())

#         for key, value in state.player1_pawns.items():

#             if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
#                 piece1 = value[0]
#                 for key, value in player_positions_on_board:
#                     piece2 = value[0]
#                     # make sure a potential neighbour is a on the board
#                     if not np.array_equal(piece2, not_on_board) and not is_hidden(state, agent_id, key):
#                         # calculate euclidean distance
#                         eucl_dist = distance.euclidean(piece1.tolist(),piece2.tolist())
#                         # if it is 1 then it is a neighbour
#                         if eucl_dist == 1.0:
#                             sum_neighbours +=1

#     if agent_id == 1:
#         # a list of all the pieces positions and pieces
#         player_positions_on_board = list(state.player2_pawns.items())

#         for key, value in state.player2_pawns.items():
#             if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
#                 piece1 = value[0]
#                 for key, value in player_positions_on_board:
#                     piece2 = value[0]
#                     # make sure a potential neighbour is a on the board
#                     if not np.array_equal(piece2, not_on_board) and not is_hidden(state, agent_id, key):
#                         # calculate euclidean distance
#                         eucl_dist = distance.euclidean(piece1.tolist(),piece2.tolist())
#                         # if it is 1 then it is a neighbour
#                         if eucl_dist == 1.0:
#                             sum_neighbours +=1

#     return sum_neighbours


# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# -------------------------------------- PART B --------------------------------------
def greedy_improved(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = neighbor_list[0]
    for neighbor in neighbor_list:
        # depth == 0 triggers smart heuristic evaluation
        curr_heuristic = smart_heuristic(neighbor[1], agent_id, 100, 0)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# -------------------------------------- PART C --------------------------------------
def minimax(curr_state, agent_id, turn, time_limit, depth):
    neighbor_list = curr_state.get_neighbors()
    # default to first one
    best_action = neighbor_list[0][0]
    current_minmax_val = float('-inf')
    current_minmax_val = float('inf') if turn else current_minmax_val

    # if turn is 0, it is our turn
    for neighbour in neighbor_list:
        state = neighbour[1]
        action_neighbour = neighbour[0]
        heuristic_val = smart_heuristic(curr_state, agent_id, time_limit, depth)
        if heuristic_val is not None:
            return heuristic_val, None
        v, _ = minimax(state, agent_id, not turn, time_limit, (depth - 1))
        if not turn:
            if v >= current_minmax_val:
                current_minmax_val = v
                best_action = action_neighbour
        else:
            if v <= current_minmax_val:
                current_minmax_val = v
                best_action = action_neighbour

    return current_minmax_val, best_action


def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    time_run_out = time.time() + time_limit * 0.99
    currently_best_action = None
    for depth in range(1, 10):
        val, action = minimax(curr_state, agent_id, 0, time_run_out, depth)
        if timeout(time_run_out):
            return currently_best_action
        currently_best_action = action


# def rb_heuristic_min_max(curr_state, agent_id, time_limit):
#     time_run_out = time.time() + time_limit*0.95

#     neighbor_list = curr_state.get_neighbors()
#     curr_heuristic = 0
#     max_heuristic = 0
#     max_neighbor = None
#     for neighbor in neighbor_list:
#         curr_heuristic = heuristic_min_max(neighbor[1], agent_id, 3)
#         print("We got hereeeeeeeee")
#         print(f'curr_heuristic is {curr_heuristic}')

#         if curr_heuristic >= max_heuristic:
#             max_heuristic = curr_heuristic
#             max_neighbor = neighbor
#             print(f'Action to take is {max_neighbor[0]}')

#         if time.time() - time_run_out < 0.0:
#             print(f'Action to take is {max_neighbor[0]}')
#             return max_neighbor[0]

#     return max_neighbor[0]

# -------------------------------------- PART D --------------------------------------
def rb_alpha_beta(curr_state, agent_id, turn, time_limit, depth, alpha, beta):
    neighbor_list = curr_state.get_neighbors()
    # default to first one
    best_action = neighbor_list[0][0]
    current_minmax_val = float('-inf')
    current_minmax_val = float('inf') if turn else current_minmax_val
    alpha = float('-inf') if turn else alpha
    beta = float('inf') if not turn else beta

    # if turn is 0, it is our turn
    for neighbour in neighbor_list:
        state = neighbour[1]
        action_neighbour = neighbour[0]
        heuristic_val = smart_heuristic(curr_state, agent_id, time_limit, depth)
        if heuristic_val is not None:
            return heuristic_val, None
        v, _ = rb_alpha_beta(state, agent_id, not turn, time_limit, (depth - 1), alpha, beta)
        if not turn:
            if v >= current_minmax_val:
                current_minmax_val = v
                best_action = action_neighbour
            if current_minmax_val >= beta:
                return current_minmax_val, best_action
            alpha = max(current_minmax_val, alpha)
        else:
            if v <= current_minmax_val:
                current_minmax_val = v
                best_action = action_neighbour
            if current_minmax_val <= alpha:
                return current_minmax_val, best_action
            beta = min(current_minmax_val, beta)

    return current_minmax_val, best_action


def alpha_beta(curr_state, agent_id, time_limit):
    time_run_out = time.time() + time_limit * 0.99
    currently_best_action = None
    for depth in range(1, 10):
        val, action = rb_alpha_beta(curr_state, agent_id, 0, time_run_out, depth, float('-inf'), float('inf'))
        if timeout(time_run_out):
            return currently_best_action
        currently_best_action = action


# -------------------------------------- PART E --------------------------------------
def is_small_pawn(pawn):
    return pawn == "S1" or pawn == "S2"


def is_eating(state, action, agent_id):
    pawns = state.player2_pawns.items() if agent_id else state.player1_pawns.items()
    for _, value in pawns:
        if action[1] == gge.cor_to_num(value[0]):
            return True
    return False


def rb_expectimax(curr_state, agent_id, turn, time_limit, depth):

    heuristic_val = smart_heuristic(curr_state, agent_id, time_limit, depth)
    if heuristic_val is not None:
        return heuristic_val, None

    neighbor_list = curr_state.get_neighbors()
    neighbour_count = len(neighbor_list)
    # default to first one
    best_action = neighbor_list[0][0]
    current_expectimax_val = float('-inf')
    current_expectimax_val = 0 if turn else current_expectimax_val
    return_val = current_expectimax_val, best_action

    # if turn is 0, it is our turn
    for child in neighbor_list:
        state = child[1]
        action_neighbour = child[0]

        v, _ = rb_expectimax(state, agent_id, not turn, time_limit, (depth - 1))
        if not turn:
            if v > current_expectimax_val:
                current_expectimax_val = v
                best_action = action_neighbour
                return_val = current_expectimax_val, best_action
        else:
            if is_small_pawn(child[0][0]) or is_eating(curr_state, action_neighbour, agent_id):
                v *= 2
                neighbour_count += 1
            current_expectimax_val += v
            return_val = current_expectimax_val / neighbour_count, None
    # if turn:
    #     return current_expectimax_val / neighbour_count, None
    # else:
    #     return current_expectimax_val, best_action
    return return_val


def expectimax(curr_state, agent_id, time_limit):
    time_run_out = time.time() + time_limit * 0.99
    currently_best_action = None
    for depth in range(1, 10):
        val, action = rb_expectimax(curr_state, agent_id, 0, time_run_out, depth)
        if timeout(time_run_out):
            return currently_best_action
        currently_best_action = action


# --------------------------------------------------------------------------
# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
