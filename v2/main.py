from statisco import StockDataFrame
from trader import Trader
import neat
import numpy as np

from functionalities import plot_live_data

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

COLUMN_INDEX_DICT = {column_name: index for index, column_name in enumerate(DATA.columns)}

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
            if traders[j].past_credit != traders[j].credit:
                ge[j].fitness += 0.1  # Reward for making a trade
            else:
                ge[j].fitness -= 0.1

            difference = traders[j].credit + (traders[j].holdings * Data_NOT_N[i][3]) - traders[j].past_credit
            if difference > 0:
                ge[j].fitness += 5 * difference  # Reward profit-making more significantly
            else:
                ge[j].fitness -= 5 * abs(difference)  # Penalize losses more harshly

            # Bonus for achieving new high scores in total capital
            total_capital = traders[j].credit + (traders[j].holdings * Data_NOT_N[i][3])
            if total_capital > STARTING_CAPITAL:
                ge[j].fitness += (total_capital - STARTING_CAPITAL) * 0.1

        for j in reversed(range(len(traders))):
            if not traders[j].check_alive():
                ge[j].fitness -= 100
                nets.pop(j)
                ge.pop(j)
                traders.pop(j)
                
    best_trader = max(traders, key=lambda trader: trader.credit + (trader.holdings * Data_NOT_N[-1][3]))
    print(f"Best trader's total capital this generation: {best_trader.credit + (best_trader.holdings * Data_NOT_N[-1][3])}")
    print(best_trader.credit_history)
    print(best_trader.holdings_history)

def evaluate_best_genome(best_genome, config):
    trader = Trader(credit=STARTING_CAPITAL)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    rows_number = len(DATA) - 1
    for i in range(10, rows_number):
        if not trader.check_alive():
            break
        decision = net.activate(np.append(DATA[i], trader.credit / 1000))

        # Apply order
        if decision[0] > 0.9:
            if trader.credit >= decision[1] * 1000:
                trader.buy(decision[1] * 1000, Data_NOT_N[i+1][3])
        elif decision[0] < 0.1:
            if trader.holdings >= (decision[1] * 1000) / Data_NOT_N[i+1][3]:
                trader.sell(decision[1] * 1000, Data_NOT_N[i+1][3])

    final_credit = trader.credit
    final_holdings_value = trader.holdings * Data_NOT_N[-1][3]
    total_capital = final_credit + final_holdings_value

    return total_capital, final_credit, final_holdings_value, trader.credit_history, trader.holdings_history

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

    number_of_generations = 150
    winner = population.run(training, number_of_generations)

    print(f"\nBest genome: \n{winner}")

    # Evaluate the best genome
    total_capital, final_credit, final_holdings_value, credit_history, holdings_history = evaluate_best_genome(winner, config)
    print(f"Total Capital: {total_capital}")
    print(f"Final Credit: {final_credit}")
    print(f"Final Holdings Value: {final_holdings_value}")
    print("Credit History:", credit_history)
    print("Holdings History:", holdings_history)

def main():
    run(CONFIG_FILE)

if __name__ == "__main__":
    main()

