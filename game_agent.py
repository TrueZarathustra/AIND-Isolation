"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


def distance(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):

    # return basic_score(game, player)
    return strategy_complex(game, player)
    # return strategy_closer_to_center(game, player)


def basic_score(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return own_moves-opp_moves


def strategy_center(game, player):
    """
    Heuristic idea:
    Assumption, that in case you have equal number of legal moves,
    it is preferrable to be closer to center

    Result:
    ID_Improved: 59,08% (average)
    Student: 66.7% (average)
    """

    xc = game.width*1.0/2
    yc = game.height*1.0/2

    x, y = game.get_player_location(player)

    return basic_score(game, player) - distance(x, y, xc, yc)*0.1


def strategy_close_or_far(game, player):
    """
    Heuristic idea:
    Move closer to opponent, when he has less moves. Otherwise run from him

    Result:
    ID_Improved: 62.5%
    Student: 64.29%
    """

    bs = basic_score(game, player)

    x1, y1 = game.get_player_location(player)
    x2, y2 = game.get_player_location(game.get_opponent(player))

    d = distance(x1, y1, x2, y2)*0.1

    if bs > 0:
        d = d*(-1)

    return bs + d*0.1


def strategy_free_field(game, player):
    """
    Heuristic idea:
    Move closer to regions with a lot of blank spaces

    Result:
    ID_Improved: 64.64%
    Student: 66,78%
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    x, y = game.get_player_location(player)
    blank = game.get_blank_spaces()

    sum_of_distances = 0
    for i in blank:
        sum_of_distances += distance(x, y, i[0], i[1])

    return float(sum_of_distances)*(-1.0)


def strategy_closer_to_center(game, player):
    """
    Heuristic idea:
    Move closer to regions with a lot of blank spaces
    """
    bs = basic_score(game, player)

    x1, y1 = game.get_player_location(player)
    x2, y2 = game.get_player_location(game.get_opponent(player))

    xc = game.width*1.0/2
    yc = game.height*1.0/2

    d1 = distance(x1, y1, xc, yc)*0.1
    d2 = distance(x2, y2, xc, yc)*0.1

    return bs + (d2-d1)*0.1


def strategy_complex(game, player):
    """
    Heuristic idea:
    Move closer to regions with a lot of blank spaces

    Result:

    """
    # score = strategy_center(game, player)
    # score = strategy_close_or_far(game, player)
    # score = strategy_free_field(game, player)
    # score = custom_score(game, player)

    # Begginning of the game
    if len(game.get_blank_spaces()) > game.width*game.height*0.7:
    # Game middle
        score = basic_score(game, player)
    elif len(game.get_blank_spaces()) > game.width*game.height*0.4:
        score = strategy_closer_to_center(game, player)
    # Game ending
    else:
        score = basic_score(game, player)

    return score


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
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

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
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

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

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if not legal_moves:
            return (-1, -1)

        next_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                depth = 0
                while True:
                    if self.method == 'minimax':
                        _, next_move = self.minimax(game, depth, True)
                    else:
                        _, next_move = self.alphabeta(game, depth, True)
                    depth += 1
            else:
                if self.method == 'minimax':
                    _, next_move = self.minimax(game, self.search_depth, True)
                else:
                    _, next_move = self.alphabeta(game, self.search_depth, True)


        except Timeout:
            # Handle any actions required at timeout, if necessary
            if next_move == (-1, -1):
                next_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

            return next_move

        # Return the best move from the last completed search iteration
        if next_move == (-1, -1):
            next_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        return next_move

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
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        move = (-1, -1)
        if maximizing_player:
            score = float("-inf")
        else:
            score = float("inf")

        if depth == 0:
            return self.score(game, self), move

        else:
            for m in legal_moves:
                score_m, move_m = self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)

                if maximizing_player:
                    if score_m > score:
                        score = score_m
                        move = m

                elif not maximizing_player:
                    if score_m < score:
                        score = score_m
                        move = m
        return score, move

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

        legal_moves = game.get_legal_moves()

        move = (-1, -1)
        score = float("-inf")

        if depth == 0:
            return self.score(game, self), move

        elif maximizing_player:
            score = float("-inf")
            for m in legal_moves:
                score_m, move_m = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)

                if score <= score_m:
                    score = score_m
                    move = m

                if score >= beta:
                    break
                alpha = max(alpha, score)

        else:
            score = float("inf")
            for m in legal_moves:
                score_m, move_m = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)

                if score >= score_m:
                    score = score_m
                    move = m

                if score <= alpha:
                    break
                beta = min(alpha, score)

        return score, move


'''

for m in legal_moves:
    score_m, move_m = self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)
    
    if maximizing_player:
        if score_m > score:
            score = score_m
            move = m

    elif not maximizing_player:
        if score_m < score:
            score = score_m
            move = m
'''