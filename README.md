# SnakeAI

The goal of this small project is to create a version of the popular snake game, and to create a simple AI using a basic reinforcement learning algorithm to play the game

The tutorial followed for the code comes from [this video](https://www.youtube.com/watch?v=L8ypSXwyBds)

## Initial Reflection (V1)

While this tutorial left me confused at time, likely because of my lack of prior knowledge on the subject, it helped me understand the basics of how models are trained on games through agents. Agents, put simply, are the bridge between the game and the model. Much like the Controllers in the Model View Controller system, they are responsible for being the bridge between the game (View) and the model (Model). 

To make everything work, the game has be to setup to not take user input, but rather accept inputs through code. These inputs will be passed in by the agent, who will basically be playing the game based on the model's predicted best move. The game and the agent in this case kept track of rewards and how many points the model had. These values were then fed into the trainer which, to me, was doing some mathemagic. The video attempted to explain what was happening, but I got lost. I assume this is something that I won't installed understand after watching a single tutorial, but rather something that I'll understand over time. 

There's still a lot I don't understand about how these models are created and the math behind them, but here's some things I noticed:
- Playing with the reward value made things messy if taken to the extremes. The current reward values are +10 for each fruit eaten and -10 for death by either collision, or not collecting a new fruit for too long (timeout). I thought of making the reward for dying -1000, but then the AI simply spun in circles until it died by timeout. I also tried setting the reward for collecting fruit +1000 while setting the death reward back to -10, but that didn't seem to change much.
- Changing the middle layer's size didn't seem to make a huge difference when increased. The default size of the network was 11 -> 256 -> 3 where the 11 inputs were whether there was danger ahead, to the right, or to the left (3), move direction booleans for whether the snake was moving left, right, up, or down (4), and booleans for whether the food was left, right, above, or below the snake (4). These inputs work at first, and the AI, from testing over 600 games on the 256 input network, hit a high score of around 80. After changing the middle layer to 512 and 1024, the results weren't much better. The main reason the AI lost was because it kept trapping itself and hitting another piece of the snake. For this reason, I imagine the biggest improvement to the AI would be to increase the number of inputs and somehow include something regarding how close it is to hitting itself, or perhaps a few layers of danger depending on where the AI is. For example, increasing the inputs to look 2 or 3 steps ahead and seeing what leads to death and what doesn't may help the AI predicts things better than just seeing 1 move ahead. Not sure if I'll implement this here, but it's definitely an important change that could be made.

## Possible Improvements (V1)
- [ ] Rewrite the code to include more inputs. Most likely inputs to help prevent the AI from running into itself in the future
- [ ] Rewrite the code to be a bit more consistent with names. The tutorial kept changing names around and had a bit of a problem with consistency. Things like game_over being referred to as done or game_over. Things like states being named next_move or final_move, but never sticking to one name. Basically make the code neater and more consistent
- [ ] Try to understand what's actually going on with the math and trainer, but this is for the future
- [ ] Mess around with different version of the model and see what works best. Different amount of inputs, middle layer size, and maybe even how the network itself does things. Changing ReLU to Tanh or Sigmoid and seeing how that performs, changing reward values and seeing how those affect it.
- [ ] Make the application multithreaded and run multiple games at once. I was looking into this and will likely do it, but it would make things much faster if there were multiple threads working on this and running multiple games. 
- [ ] Store all the necessary data for the graph to be accurate. The graph doesn't store total score, so the graph will start at 0 total score over whatever amount of games is stored, and it will compute a completely wrong average per game amount. For example, my current model has 677 games with an average of 70 score per game. After the first game, the average score will be 70/677 because the total score was not previously stored. 

## Versions
### V1
This is the initial version of the game and AI as done in the initial tutorial found [here](https://www.youtube.com/watch?v=L8ypSXwyBds). The above reflections and possible changes have been taken into consideration for the newer version of this AI. Some initial changes were left in here which may or may not have done anything. One example is `SharedResources` in the `helper.py`. These changes don't make a difference anyway, so ignore them. 

This version of the AI uses Q Learning and only 1 snake.

### V2
This is the first version created without following a tutorial, but instead created by making changes that I thought would improve the AI. Some improvements have been discussed above in the V1 [possible improvements](#possible-improvements) section.

This version of the AI uses multiple snakes and an evolution based AI. This approach starts with a set number of snakes, n, which will all initially be randomize. These snakes will all do their thing for each generation, and the best performers, selected based on a couple different factors, will be used as the base models for the next generation. 

## Change Logs
This will keep track of things changed from version to version
### V1
Nothing major was changed from the original tutorial

### V2
The goal of this version was to allow multiple snakes to play at once as well as increasing the number of inputs to allow for a more advanced AI.
- Updated the game to handle `n_snakes`
- Updated the inputs
  - Initial inputs:
    ```python
    Total 11
    Is there a danger straight ahead - 0 or 1 - bool
    Is there a danger to the right - 0 or 1 - bool
    Is there a danger to the left - 0 or 1 - bool
    Is snake moving left - 0 or 1 - bool
    Is snake moving right - 0 or 1 - bool
    Is snake moving up - 0 or 1 - bool
    Is snake moving down - 0 or 1 - bool
    Is the food to the left of us - 0 or 1 - bool
    Is the food to the right of us - 0 or 1 - bool
    Is the food above us - 0 or 1 - bool
    Is the food below us - 0 or 1 - bool
    ```
  - New inputs:
    ```
    Total 92
    9x9 grid around snake's head with values -1 for danger, 0 for no danger, and 1 for food - 81 inputs here
    11 inputs from previous version
    ```
    The goal of increasing the snake's view is to help it not run into itself.
- More optimized runs, haven't measured this yet
- Multi threading speedups, so far only basic operations have been done this way, which may not even speed up overall process
- No GUI version for faster training, save model, use model on GUI version for visualization