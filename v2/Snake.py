class Snake:
    def __init__(self, direction, head, snake, score, food, gameOver, model):
        self.direction = direction
        self.head = head
        self.snake = snake
        self.score = score
        self.food = food
        self.gameOver = gameOver
        self.model = model
        self.frameIterations = 0
        # Used to keep track of how it died
        # 0 Lazy, didn't get any more fruit and just died bc of multiplier death
        # 1 Wall
        # 2 Hit itself
        self.death = None
        self.finalLength = None
    
    def getDirection(self):
        return self.direction

    def setDirection(self, newDir):
        self.direction = newDir

    def getHead(self):
        return self.head

    def setHead(self, newHead):
        self.head = newHead

    def getSnake(self):
        return self.snake

    def setSnake(self, newSnake):
        self.snake = newSnake

    def getScore(self):
        return self.score

    def setScore(self, newScore):
        self.score = newScore

    def getFood(self):
        return self.food

    def setFood(self, newFood):
        self.food = newFood
    
    def getGameOver(self):
        return self.gameOver

    def setGameOver(self, newGameOver):
        self.gameOver = newGameOver

    def getModel(self):
        return self.model

    def setModel(self, newModel):
        self.model = newModel

    def getFrameIterations(self):
        return self.frameIterations

    def setFrameIterations(self, newIterations):
        self.frameIterations = newIterations
    
    def getDeath(self):
        return self.death
    
    def setDeath(self, newDeath):
        self.death = newDeath
    
    def getFinalLength(self):
        return self.finalLength

    def setFinalLength(self, newFinalLength):
        self.finalLength = newFinalLength
