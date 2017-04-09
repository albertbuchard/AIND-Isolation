"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import numpy as np
import re


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def matrix_to_string(A):
    '''
    Returns a string value for the matrix A
    '''
    A = np.asarray(A).flatten()
    V = '-'.join(str(e) for e in A)
    return V


def is_symetry_string_in_list(needle, haystack):
    '''
    Takes a string representation of a square matrix as a needle and
    Returns true if any of the 8 symetries of the specified needle is in the haystack
    '''
    try:
        symetries = string_symetries(needle)
    except ValueError:
        raise
    return np.any([e in symetries for e in haystack.keys()])


def value_for_symetry_in_list(needle, haystack):
    '''
    Returns the value of the first symetry met in the haystack dictionary
    '''
    try:
        symetries = string_symetries(needle)
    except ValueError:
        raise

    value = None
    for key in haystack.keys():
        if key in symetries:
            value = haystack[key]
            break

    # Did not find a value
    return value


def exchange_matrix(dim):
    '''
    Returns an exchange matrix of size dim*dim
    '''
    return np.reshape(np.array([0 if (e != (e // dim) * dim + (dim - (e // dim)) - 1)
                                else 1
                                for e in range(dim**2)]), (dim, dim))


def string_symetries(A):
    '''
    From a string, an array or a matrix, returns the 8 string representation of the
    square symetries.
    '''
    if not isinstance(A, np.matrix):
        if isinstance(A, str):
            A = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', A)
            if len(A) == 1:
                A = A[0]
        ndim = np.sqrt(len(A))
        try:
            A = np.matrix([int(e) for e in A])
        except ValueError:
            raise ValueError("Could not convert data to an matrix.")

        if int(ndim) != ndim:
            raise ValueError('A is not a matrix nor convertible to a square matrix')

        A = np.reshape(A, (ndim, ndim))

    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not a square matrix')

    # Build the height square synmetries
    S = exchange_matrix(A.shape[0])
    At = A.T
    symetries = [A, At, At.dot(S), S.dot(A), S.dot(At), A.dot(S), S.dot(At).dot(S), S.dot(A).dot(S)]

    # Convert matrix to dictionary key strings
    strings = [matrix_to_string(symetry) for symetry in symetries]

    '''
    print("1 is \n ", A)
    print("2 is \n ", At)
    print("3 is \n ", At.dot(S))
    print("4 is \n ", S.dot(A))
    print("5 is \n ", S.dot(At))
    print("6 is \n ", A.dot(S))
    print("7 is \n ", S.dot(At).dot(S))
    print("8 is \n ", S.dot(A).dot(S))
    '''

    return strings


def add_symetric_keys(key, value, hashTable):
    '''
    Copy the value to the 7 other symetric keys in the hashTable
    '''
    if not isinstance(hashTable, dict):
        raise ValueError("hashTable is not an instance of dictionary")
    hashTable.update({e: value for e in string_symetries(key)})
    return hashTable


def custom_stored_score(game, player, heuristic=custom_score):

    state = game._board_state[:-3]
    state[game._board_state[-1]] = 2
    state[game._board_state[-2]] = 3

    score = value_for_symetry_in_list(state, player.visited_scores)

    if score is None:
        score = heuristic(game, player)
        player.visited_scores = add_symetric_keys(state, score, player.visited_scores)

    return score


def custom_stored_score_rollout(game, player, heuristic=custom_score):

    state = game._board_state[:-3]
    state[game._board_state[-1]] = 2
    state[game._board_state[-2]] = 3
    dim = int(np.sqrt(len(state)))
    state = np.reshape(state, (dim, dim), order='F')

    score = value_for_symetry_in_list(state, player.visited_scores)

    if score is None:
        score = heuristic(game, player)
        player.visited_scores = add_symetric_keys(state, score, player.visited_scores)
    else:
        # start fast rollout
        print(score)
        rolls = []
        legals = game.get_legal_moves(player)
        if len(legals) > 0:
            idp = -1
            ido = -2
            vp = 2
            if player == game._player_2:
                idp = -2
                ido = -1
                vp = 3
            state[game._board_state[idp]] = 1
            keys = []
            ngame = game.copy()
            propagate = 0
            for d in range(30):
                sample = np.random.randint(0, len(legals))
                move = legals[sample]

                ngame = ngame.forecast_move(move)
                idx = move[0] + move[1] * ngame.height
                state = game._board_state[:-3]
                state[game._board_state[-2]] = 2
                state[game._board_state[-2]] = 3
                state[idx] = vp
                if player == ngame._inactive_player:
                    keys.append(matrix_to_string(state))

                score = heuristic(ngame, player)
                legals = ngame.get_legal_moves(ngame._active_player)
                if not legals:

                    if player == ngame._inactive_player:
                        propagate = float("inf")

                    if player == ngame._active_player:
                        propagate = float("-inf")

                    break

            if propagate != 0:
                for key in keys:
                    player.visited_scores = add_symetric_keys(key, propagate, player.visited_scores)

                return propagate

    return score


def custom_score(game, player, agressiveness=3):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    number_of_moves = game.get_legal_moves(player)
    oponent_moves = game.get_legal_moves(game.get_opponent(player))
    score = float(len(number_of_moves) - (agressiveness * len(oponent_moves)))
    return score


def custom_score_blanks(game, player):
    st = game._board_state[:-3]
    dim = int(np.sqrt(len(st)))
    st = np.reshape(st, (dim, dim), order='F')
    loc = game.get_player_location(player)
    if loc is None:
        return 0
    minr = max(0, loc[0] - 2)
    minc = max(0, loc[1] - 2)
    maxr = min(dim - 1, loc[0] + 2)
    maxc = min(dim - 1, loc[1] + 2)
    w = []
    return len([w.append((r, c)) if st[r, c] == 0 else 0 for r in range(minr, maxr + 1) for c in range(minc, maxc + 1)])


def custom_score_blanks_agressive(game, player):
    st = game._board_state[:-3]
    dim = int(np.sqrt(len(st)))
    st = np.reshape(st, (dim, dim), order='F')

    # Get number of blank cell around the player
    loc = game.get_player_location(player)
    if loc is None:
        return 0
    minr = max(0, loc[0] - 2)
    minc = max(0, loc[1] - 2)
    maxr = min(dim - 1, loc[0] + 2)
    maxc = min(dim - 1, loc[1] + 2)
    w = []
    number_of_blank_player = len([w.append((r, c)) if st[r, c] == 0 else 0 for r in range(minr, maxr + 1) for c in range(minc, maxc + 1)])

    # Get number of blank cell around the opponent
    loc = game.get_player_location(game.get_opponent(player))
    if loc is None:
        return 0
    minr = max(0, loc[0] - 2)
    minc = max(0, loc[1] - 2)
    maxr = min(dim - 1, loc[0] + 2)
    maxc = min(dim - 1, loc[1] + 2)
    w = []
    number_of_blank_opponent = len([w.append((r, c)) if st[r, c] == 0 else 0 for r in range(minr, maxr + 1) for c in range(minc, maxc + 1)])

    return number_of_blank_player - number_of_blank_opponent


def custom_score_generator(agressiveness=1):
    def custom(game, player):
        return custom_score(game, player, agressiveness)

    return custom


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10., agressiveness=None):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.move_history = []
        self.open_moves = []
        self.agressiveness = agressiveness
        self.visited_scores = {}

    def get_fill_square_sequence(self, game):
        # TODO may be interesting open book
        j = np.array([0, 2, 1, 0, 2, 0, 1, 2, 0, 1])
        i = np.array([0, 1, -1, 1, 0, -1, 1, -1, -2, 0])
        offsets = [np.array((i, j)), np.array((i, -j)), np.array((-i, j)),
                   np.array((-j, i)), np.array((j, i)), np.array((-j, -i)), np.array((j, -i))]
        location = np.resize(np.array(game.get_player_location()), (2, 1))
        sequence = [location + offset for offset in offsets]
        [[(seq[0, i], seq[1, i]) for i in range(np.shape(seq)[1])] for seq in sequence]

    def has_moved_to(self, move):
        '''
        Add a move to the move history for the player
        '''
        self.move_history.append(list(move))

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.


        """
        # TODOs
        # -----
        # Perform any required initializations, including selecting an initial CHANGED
        # move from the game board (i.e., an opening book) TODO finish square sequence
        # returning immediately if there are no legal moves CHANGED
        self.time_left = time_left

        if game.get_player_location(self) == game.NOT_MOVED:
            # print('NOT MOVED !')
            # this is the first move
            center_position = (int(np.ceil(game.width / 2)), int(np.ceil(game.height / 2)))
            if game.get_player_location(game.get_opponent(self)) == game.NOT_MOVED or game.move_is_legal(center_position):
                # We are player one or center is free : pick center
                # print('CENTER POSITION ', center_position)
                return center_position
            else:
                return tuple(np.add(center_position, (1, 0)))

        # TODO implement the open book
        '''
        elif len(self.move_history) < 2 and not len(self.open_moves):
            # first moves, try to fit a square sequence
            # if possible
            self.open_moves = self.get_fill_square_sequence()

        if len(self.open_moves):
            # we are currently attempting to perform open book sequence do not perform search until blocked
            # check if next position on sequence is possible
            # if yes perform and increment sequence index
            next_open_position = self.open_moves.pop(0)
            if game.move_is_legal(next_open_position):
                return next_open_position
            else:
                # stop the open sequence
                self.open_moves = []
        '''
        # Not at the beggining, not doing the open -- start performing the search
        legal_moves = game.get_legal_moves(self)

        if not legal_moves:
            return (-1, -1)

        score = float('-inf')
        move = (-1, -1)
        try:
            maximizing_player = True

            if self.iterative:
                depth = 1
                while True:
                    if self.method == 'minimax':
                        score, move = self.minimax(game, depth, maximizing_player)
                    elif self.method == 'alphabeta':
                        score, move = self.alphabeta(game, depth, float("-inf"), float("inf"), maximizing_player)

                    depth += 1
            else:
                if self.method == 'minimax':
                    score, move = self.minimax(game, self.search_depth, maximizing_player)
                elif self.method == 'alphabeta':
                    score, move = self.alphabeta(game, self.search_depth, float("-inf"), float("inf"), maximizing_player)

        except Timeout:
            return move

        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.


                function MINIMAX-DECISION(state) returns an action
                 return arg max a ∈ ACTIONS(s) MIN-VALUE(RESULT(state, a))

                function MAX-VALUE(state) returns a utility value
                 if TERMINAL-TEST(state) the return UTILITY(state)
                 v ← −∞
                 for each a in ACTIONS(state) do
                   v ← MAX(v, MIN-VALUE(RESULT(state, a)))
                 return v

                function MIN-VALUE(state) returns a utility value
                 if TERMINAL-TEST(state) the return UTILITY(state)
                 v ← ∞
                 for each a in ACTIONS(state) do
                   v ← MIN(v, MAX-VALUE(RESULT(state, a)))
                 return v
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        player = game.active_player
        # print("At depth ", depth, " ---- ", player, " -- last move p1", self.move_history[-1])
        legal_moves = game.get_legal_moves(player)

        # if no legal moves return utility
        if not legal_moves:
            if player == self:
                utility = float("-inf")
            else:
                utility = float("+inf")
            return utility, (-1, -1)

        # check if we are at a leaf (depth == 0)
        if depth == 0:
            # we are at a leaf return the board score
            return self.score(game, self), (-1, -1)
        else:
            # not at a leaf - use recursion alternating min and max to search the tree
            if maximizing_player:
                results = [(self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)[0], m) for m in legal_moves]
                # print('RESULTS : ', results)
                # print('MAX : ', max(results))
                return max(results)
            else:
                return min([(self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)[0], m) for m in legal_moves])

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        player = game.active_player
        # print("At depth ", depth, " ---- ", player, " -- last move p1", self.move_history[-1])
        legal_moves = game.get_legal_moves(player)

        if not legal_moves:
            if player == self:
                utility = float("-inf")
            else:
                utility = float("+inf")
            return utility, (-1, -1)

        if depth == 0:
            # we are at a leaf - return board score
            return self.score(game, self), (-1, -1)
        else:
            if maximizing_player:
                best_score = float('-inf')
                best_move = (-1, -1)
                # print(' beta ', beta)

                # go through all the moves but this time using a for loop in order to prune
                for m in legal_moves:
                    score, move = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)

                    # check if the score is above beta for pruning
                    if score >= beta:
                        # score is above beta, this branch will not be selected 2 levels above
                        # can be prunned
                        # print('Score : ', score)
                        # print('move : ', move)
                        return score, m
                    else:
                        alpha = max(alpha, score)
                        if score > best_score:
                            best_score = score
                            best_move = m
                return best_score, best_move

            else:
                best_score = float('+inf')
                best_move = (-1, -1)
                # print(' alpha ', alpha)
                for m in legal_moves:
                    score, move = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)
                    # print('Score : ', score, ' alpha ', alpha)
                    if score <= alpha:
                        # print('PRUNED!')
                        return score, m
                    else:
                        beta = min(beta, score)
                        if score < best_score:
                            best_score = score
                            best_move = m
                return best_score, best_move
