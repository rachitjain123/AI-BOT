from copy import deepcopy
import random
from sys import maxsize
import sys
import traceback
from time import time
class bot:
    def __init__(self):
        self.startTime = 0
        self.timeLimit = 15.5
        self.default_depth = 3
        self.heuristicDict = {}
        self.infinity = 100000000
        self.blk_won_p = 0 # how many blocks won by player
        self.blk_won_o = 0 # how many blocks won by opponent
        self.is_bonus = 0
        self.Util_Matrix = [[1,0,0,0,0],[3,0,0,0,0],[9,0,0,0,0],[27,0,0,0,0],[81,0,0,0,0]]

    def getBlockStatus(self,block):
        row_pos = [[0,0],[0,1],[0,2],[0,3]] #initial row index
        for i,k in row_pos:
            if block[i][k]==block[i+1][k]==block[i+2][k]==block[i+3][k] and block[i][k] in (1,2): #columns
                return block[i][k]

        col_pos = [[0,0],[1,0],[2,0],[3,0]] #initial col index
        for i,k in col_pos:
            if block[i][k]==block[i][k+1]==block[i][k+2]==block[i][k+3] and block[i][k] in (1,2): #columns
                return block[i][k]

        diamond_pos = [[1,0],[2,0],[1,1],[2,1]] #Left positions of diamond
        for i,k in diamond_pos:
            if block[i][k] == block[i-1][k+1] == block[i+1][k+1] == block[i][k+2] and block[i][k] in (1,2):
                return block[i][k]

        for i in range(4):
            for j in range(4):
                if block[i][j] == 0:
                    return 0
        return 3

    def count(self, i, j, block):

        row_ply = 0
        row_opp = 0
        col_ply = 0
        col_opp = 0
        dia_ply = 0
        dia_opp = 0        
        score = 0

        new_block = deepcopy(block)
        for row in range(4):
            if new_block[i][row] == 1:
                row_ply += 1
            if new_block[i][row] == 2:
                row_opp += 1

        score += self.Util_Matrix[row_ply][row_opp]

        for col in range(4):
            if new_block[col][j] == 1:
                col_ply += 1
            if new_block[col][j] == 2:
                col_opp += 1

        score += self.Util_Matrix[col_ply][col_opp]

        diamond1 = [[1,0],[0,1],[2,1],[1,2]]
        if [i,j] in diamond1:
            for k,l in diamond1:
                if new_block[k][l] == 2:
                    dia_opp += 1
                if new_block[k][l] == 1:
                    dia_ply += 1
            score += self.Util_Matrix[dia_ply][dia_opp]

        dia_ply = 0
        dia_opp = 0

        diamond2 = [[2,0],[1,1],[3,1],[2,2]]
        if [i,j] in diamond2:
            for k,l in diamond2:
                if new_block[k][l] == 1:
                    dia_ply += 1
                if new_block[k][l] == 2:
                    dia_opp += 1
            score += self.Util_Matrix[dia_ply][dia_opp]

        dia_ply = 0
        dia_opp = 0

        diamond3 = [[1,1],[0,2],[2,2],[1,3]]
        if [i,j] in diamond3:
            for k,l in diamond3:
                if new_block[k][l] == 1:
                    dia_ply += 1
                if new_block[k][l] == 2:
                    dia_opp += 1
            score += self.Util_Matrix[dia_ply][dia_opp]

        dia_ply = 0
        dia_opp = 0

        diamond4 = [[2,1],[1,2],[3,2],[2,3]]
        if [i,j] in diamond4:
            for k,l in diamond4:
                if new_block[k][l] == 1:
                    dia_ply += 1
                if new_block[k][l] == 2:
                    dia_opp += 1
            score += self.Util_Matrix[dia_ply][dia_opp]

        return score

    def getBlockScore(self,block):
        blk_list  = deepcopy(block)
        block = tuple([tuple(block[i]) for i in range(4)])
        if block not in self.heuristicDict:
            blockStat = self.getBlockStatus(block)
            if blockStat == 1:
                self.heuristicDict[block] = 100
            elif blockStat == 2 or blockStat == 3:
                self.heuristicDict[block] = 0 #Check whether to keep 0.0 or not
            else:
                best = -100000
                moves = []
                playBlock = blk_list
                opnplayBlock = deepcopy(playBlock)

                for i in range(4):
                    for j in range(4):
                        if block[i][j] == 0:
                            moves.append((i, j))
                        if opnplayBlock[i][j]:
                            opnplayBlock[i][j] = 3 - opnplayBlock[i][j]

                for move in moves:
                    ans = 1 + self.count(move[0],move[1],playBlock)
                    if ans > best:
                        best = ans
                    self.heuristicDict[block] = best
        return self.heuristicDict[block]

    def lineScore(self, line, blockProb, revBlockProb, currentBlockStatus):
        positiveScore = []
        negativeScore = []
        for x in line:
            if currentBlockStatus[x[0]][x[1]] == 3:
                return 0
            positiveScore.append(blockProb[x[0]][x[1]])
            negativeScore.append(revBlockProb[x[0]][x[1]])

        return positiveScore[0] * positiveScore[1] * positiveScore[2] * positiveScore[3] -  negativeScore[0] * negativeScore[1] * negativeScore[2] * negativeScore[3]

    def getBoardScore(self, board, currentBoard, currentBlockStatus):
        terminalStat, terminalScore = self.terminalCheck(currentBlockStatus)
        if terminalStat:
            return terminalScore
        revCurrenBoard = deepcopy(currentBoard)
        for i in range(16):
            for j in range(16):
                if revCurrenBoard[i/4][j/4][i%4][j%4]:
                    revCurrenBoard[i/4][j/4][i%4][j%4] = 3 - revCurrenBoard[i/4][j/4][i%4][j%4]

        blockProb = [[self.getBlockScore(currentBoard[i][j]) for i in range(4)] for j in range(4)]
        revBlockProb = [[self.getBlockScore(revCurrenBoard[i][j]) for i in range(4)] for j in range(4)]

        boardScore = []

        for i in range(4):
            line = [(i,0),(i,1),(i,2),(i,3)]
            val = self.lineScore(line, blockProb, revBlockProb, currentBlockStatus)
            if val == self.infinity:
                return self.infinity
            if val == -self.infinity:
                return -self.infinity
            boardScore.append(val)

            line = [(0,i),(1,i),(2,i),(3,i)]
            val = self.lineScore(line, blockProb, revBlockProb, currentBlockStatus)
            if val == self.infinity:
                return self.infinity
            if val == -self.infinity:
                return -self.infinity
            boardScore.append(val)

        diamond_pos = [[1,0],[2,0],[1,1],[2,1]]
        for i,k in diamond_pos:
            line = [[i,k], [i-1,k+1], [i+1,k+1] , [i,k+2]]
            val = self.lineScore(line, blockProb, revBlockProb, currentBlockStatus)
            if val == self.infinity:
                return self.infinity
            if val == -self.infinity:
                return -self.infinity
            boardScore.append(val)
            
        cnt1 = sum(blocks.count(self.player_symbol) for blocks in board.block_status)
        cnt2 = sum(blocks.count(self.opponent_symbol) for blocks in board.block_status)
        if self.blk_won_p < cnt1 and self.is_bonus == 1:
            print "inside bonus move"
            return self.infinity
       
        return sum(boardScore)

    def map_fun(self,currentBoard, formattedBoard, formattedBlockStatus):
            
        for i in range(16):
            for j in range(16):
                if currentBoard.board_status[i][j] == self.player_symbol:
                    formattedBoard[i/4][j/4][i%4][j%4] = 1
                elif currentBoard.board_status[i][j] == '-':
                    formattedBoard[i/4][j/4][i%4][j%4] = 0
                else:
                    formattedBoard[i/4][j/4][i%4][j%4] = 2

        for i in range(4):
            for j in range(4):
                if currentBoard.block_status[i][j] == self.player_symbol:
                    formattedBlockStatus[i][j] = 1
                elif currentBoard.block_status[i][j] == self.opponent_symbol:
                    formattedBlockStatus[i][j] = 2
                elif currentBoard.block_status[i][j] == 'd':
                    formattedBlockStatus[i][j] = 3
                else:
                    formattedBlockStatus[i][j] = 0

    def move(self,currentBoard,old_move,flag):
        try:
            if old_move == (-1, -1):
                return (4, 5)
            self.startTime = time()
            if flag == 'x':
                self.player_symbol = 'x'
                self.opponent_symbol = 'o'
            else:
                self.player_symbol = 'o'
                self.opponent_symbol = 'x'

            # check_bonus(currentBoard,old_move)

            self.blk_won_p = sum(blocks.count(self.player_symbol) for blocks in currentBoard.block_status)
            self.blk_won_o = sum(blocks.count(self.opponent_symbol) for blocks in currentBoard.block_status)

            temp_board = deepcopy(currentBoard)

            formattedBlockStatus = [[0 for i in range(4)] for j in range(4)]
            formattedBoard = [[[[0 for i in range(4)] for j in range(4)] for k in range(4)] for l in range(4)]

            self.map_fun(currentBoard, formattedBoard, formattedBlockStatus)

            for i in range(1,20):
                depth = i
                self.is_bonus =i 
                uselessScore,best_move, retDepth = self.ab_minmax(temp_board, formattedBoard,formattedBlockStatus,-100000000000000000,100000000000000000, self.player_symbol, old_move, depth)
                if (time() - self.startTime) < self.timeLimit:
                    print i, best_move, uselessScore
                    if uselessScore == self.infinity and depth == 1:
                        print "sdfsd"
                        nextMove = best_move
                        break
                    else:
                        nextMove = best_move
                else:
                    break       

            try:
                return nextMove
            except:
                print 'Tiraceback printing', sys.exc_info()
                print traceback.format_exc()
                valid_moves = currentBoard.find_valid_move_cells(old_move)
                random.shuffle(valid_moves)
                nextMove = valid_moves[0]
                return nextMove

        except Exception as e:
            print 'Exception occurred ', e
            print 'Traceback printing ', sys.exc_info()
            print traceback.format_exc()

    def undo_move(self, board, move, tempBoard,tempBlockStatus):
        board.board_status[move[0]][move[1]] = '-'
        board.block_status[move[0]/4][move[1]/4] = '-'
        tempBoard[move[0]/4][move[1]/4][move[0]%4][move[1]%4] = 0
        tempBlockStatus[move[0]/4][move[1]/4] = self.getBlockStatus(tempBoard[move[0]/4][move[1]/4])

    def exec_move(self, board, prevMove, move, tempBoard,tempBlockStatus, flag):
        board.update(prevMove, move, flag)
        if flag == self.player_symbol:
            tempBoard[move[0]/4][move[1]/4][move[0]%4][move[1]%4] = 1
        else:
            tempBoard[move[0]/4][move[1]/4][move[0]%4][move[1]%4] = 2
        tempBlockStatus[move[0]/4][move[1]/4] = self.getBlockStatus(tempBoard[move[0]/4][move[1]/4])


    def terminalCheck(self, currentBlockStatus):
        terminalStat = self.getBlockStatus(currentBlockStatus)
        blockCount = 0
        Utility_Matrix =[0,self.infinity,-self.infinity]
        if terminalStat == 1 or terminalStat ==2 :
            return True,Utility_Matrix[terminalStat]
        elif terminalStat == 0:
            return False,Utility_Matrix[0]
        else:
            for i in range(4):
                for j in range(4):
                    if currentBlockStatus[i][j] == 1:
                            blockCount += 10
                    elif currentBlockStatus[i][j] == 2:
                        blockCount -= 10
            return True, blockCount

    def ab_minmax(self,board, currentBoard, currentBlockStatus, alpha, beta, flag, prevMove, depth):
        terminalStat, terminalScore = self.terminalCheck(currentBlockStatus)
        if terminalStat:
            return terminalScore, (), 0

        if time() - self.startTime > self.timeLimit:
            return 0, (), 0

        if depth<=0:
            utility = self.getBoardScore(board, currentBoard, currentBlockStatus)
            return utility, (), 0

        valid_moves = board.find_valid_move_cells(prevMove)
        random.shuffle(valid_moves)

        if len(valid_moves) == 0:
            utility = self.getBoardScore(board, currentBoard, currentBlockStatus)
            return utility, (), 0

        bestMove = ()
        bestDepth = 100
        if flag == self.player_symbol:
            v = -1000000000
            for move in valid_moves:

                self.exec_move(board,prevMove, move, currentBoard, currentBlockStatus, flag)
                childScore, childBest, childDepth = self.ab_minmax(board, currentBoard,currentBlockStatus, alpha, beta, self.opponent_symbol, move, depth-1)
                self.undo_move(board, move, currentBoard, currentBlockStatus)

                if childScore >= v:
                    if v < childScore:# or bestDepth > childDepth:
                        v = childScore
                        bestMove = move
                        bestDepth = childDepth
                alpha = max(alpha, v)
                if (time() - self.startTime) > self.timeLimit:
                        return 0, (), 0
                
                if alpha >= beta:
                    break

            return v, bestMove, bestDepth+1
        else:
            v = 1000000000
            for move in valid_moves:

                self.exec_move(board, prevMove, move,currentBoard, currentBlockStatus, flag)
                childScore, childBest, childDepth = self.ab_minmax(board, currentBoard, currentBlockStatus, alpha, beta, self.player_symbol, move, depth-1)
                self.undo_move(board, move, currentBoard, currentBlockStatus)

                if childScore <= v:
                    if v > childScore:# or bestDepth > childDepth:
                        v = childScore
                        bestMove = move
                        bestDepth = childDepth
                beta = min(beta, v)
                if (time() - self.startTime) > self.timeLimit:
                        return 0, (), 0
                if alpha >= beta:
                    break

            return v, bestMove, bestDepth+1
