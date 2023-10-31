import matplotlib.pyplot as plt
from IPython import display
import threading
model_folder_path = "./v1/model"

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

class ShareResources:
    def __init__(self):
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.record = 0
        self.lock = threading.Lock()

    def updateValues(self, score, n_games, model):
        with self.lock:
            self.plot_scores.append(score)
            self.total_score += score
            mean_score = self.total_score / n_games
            self.plot_mean_scores.append(mean_score)
            if score > self.record:
                self.record = score
                model.save(n_games)

    def getPlotPoints(self):
        return self.plot_scores, self.plot_mean_scores
            