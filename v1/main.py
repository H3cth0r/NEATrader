from statisco import StockDataFrame
from trader import Trader
import neat
import numpy as np

# Constants
STARTING_CAPITAL = 100
DATA = StockDataFrame(ticker="MSFT", start="2022-01-01", end="2023-12-31", interval="1d")
CONFIG_FILE = "./config-feedforward"

# Apply calculations to the dataset
DATA.calculate(close_returns=True)
DATA.calculate(sma=True, interval=3)
DATA.calculate(ema=True, interval=3, smooth=3)
DATA.calculate(wma=True, interval=3)
DATA.calculate(atr=True, interval=3)
DATA.calculate_MACD(short_window=12, long_window=26, signal_window=9)
Data_NOT_N = DATA.copy().to_numpy()
DATA = DATA.normalize(inplace=False)

def training(genomes, config):
    traders = []
    nets = []
    ge = []

    for _, genome in genomes:
        trader = Trader(credit=STARTING_CAPITAL)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        traders.append(trader)
        nets.append(net)
        ge.append(genome)
        
    rows_number = len(DATA) - 1
    for i in range(10, rows_number):
        for j in range(len(traders)):
            if not traders[j].check_alive():
                continue
            decision = nets[j].activate(np.append(DATA[i], traders[j].credit / 1000))

            # Apply order
            if decision[0] > 0.9:
                if traders[j].credit >= decision[1] * 1000:
                    traders[j].buy(decision[1] * 1000, Data_NOT_N[i+1][3])
                else:
                    ge[j].fitness -= 10  # Penalize for trying to buy without enough credit
            elif decision[0] < 0.1:
                if traders[j].holdings >= (decision[1] * 1000) / Data_NOT_N[i+1][3]:
                    traders[j].sell(decision[1] * 1000, Data_NOT_N[i+1][3])
                else:
                    ge[j].fitness -= 10  # Penalize for trying to sell without enough holdings

            # Apply fitness function
            difference = traders[j].credit - traders[j].past_credit
            if traders[j].credit > traders[j].past_credit:
                ge[j].fitness += 5 * difference  # Reward profit-making more significantly
            else:
                ge[j].fitness -= 5 * abs(difference)  # Penalize losses more harshly

            # Bonus for achieving new high scores in credit
            if traders[j].credit > STARTING_CAPITAL:
                ge[j].fitness += (traders[j].credit - STARTING_CAPITAL) * 0.1

        for j in reversed(range(len(traders))):
            if not traders[j].check_alive():
                ge[j].fitness -= 100
                nets.pop(j)
                ge.pop(j)
                traders.pop(j)
                
    best_trader = max(traders, key=lambda trader: trader.credit)
    print(f"Best trader's credit this generation: {best_trader.credit}")
    print(best_trader.credit_history)
    print(best_trader.holdings_history)

def run(config_path):
    config = neat.Config(
                    neat.DefaultGenome, 
                    neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,
                    neat.DefaultStagnation,
                    config_path
            )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    number_of_generations = 250
    winner = population.run(training, number_of_generations)

    print(f"\nBest genome: \n{winner}")

def main():
    run(CONFIG_FILE)

if __name__ == "__main__":
    main()
