# dummy.py
import numpy as np
import pandas as pd
from statisco import StockDataFrame
from trader import Trader
import neat

from functionalities import plot_live_data

# Constants
STARTING_CAPITAL = 1000
STARTING_HOLDINGS = 0
N_DAYS_BATCH = 4  # Set your batch size in days here
N_TEST_BATCHES = 2  # Set the number of test batches here

# Load Data
DATA = pd.read_csv("../resources/MSFT_intraday_1min.csv")
DATA['Date'] = pd.to_datetime(DATA['Date'])
DATA["Adj Close"] = DATA["Close"]
DATA.set_index('Date', inplace=True)

DATA = DATA.sort_values(by="Date")
DATA = StockDataFrame(data=DATA)
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

# Calculate the number of rows per day
rows_per_day = DATA.resample('D').size()
print(rows_per_day)

# Normalize the data
DATA = DATA.normalize(inplace=False)

# Create batches based on days
def create_batches(data, n_days):
    batch_size = rows_per_day.iloc[:n_days].sum()
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return batches

# Split data into training and testing batches
train_batches = create_batches(DATA, N_DAYS_BATCH * (len(rows_per_day) // (N_DAYS_BATCH + N_TEST_BATCHES)))
test_batches = create_batches(DATA, N_DAYS_BATCH * N_TEST_BATCHES)

actual_train_batches = create_batches(Data_NOT_N, N_DAYS_BATCH * (len(rows_per_day) // (N_DAYS_BATCH + N_TEST_BATCHES)))
actual_test_batches = create_batches(Data_NOT_N, N_DAYS_BATCH * N_TEST_BATCHES)

batch_counter = 0

def training(genomes, config):
    global batch_counter
    traders = []
    nets = []
    ge = []

    global STARTING_CAPITAL, STARTING_HOLDINGS

    for _, genome in genomes:
        trader = Trader(credit=STARTING_CAPITAL, holdings=STARTING_HOLDINGS)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        traders.append(trader)
        nets.append(net)
        ge.append(genome)

    # Get the current batch
    current_normalized_batch = train_batches[batch_counter]
    current_actual_batch = actual_train_batches[batch_counter]
    batch_counter = (batch_counter + 1) % len(train_batches)
    
    rows_number = len(current_normalized_batch) - 1
    for i in range(10, rows_number):
        for j in range(len(traders)):
            if not traders[j].check_alive():
                continue
            decision = nets[j].activate(np.append(current_normalized_batch[i], traders[j].credit / 10000))

            # Apply order
            if decision[0] > 0.9:
                if traders[j].credit >= decision[1] * 10000:
                    traders[j].buy(decision[1] * 10000, current_actual_batch[i+1][3])
                else:
                    ge[j].fitness -= 10  # Penalize for trying to buy without enough credit
            elif decision[0] < 0.1:
                if traders[j].holdings >= (decision[1] * 10000) / current_actual_batch[i+1][3]:
                    traders[j].sell(decision[1] * 10000, current_actual_batch[i+1][3])
                else:
                    ge[j].fitness -= 10  # Penalize for trying to sell without enough holdings

            # Apply fitness function
            total_capital = traders[j].credit + (traders[j].holdings * current_actual_batch[i][3])
            ge[j].fitness = total_capital  # Directly set fitness to total capital

            # Bonus for achieving new high scores in total capital
            if total_capital > STARTING_CAPITAL:
                ge[j].fitness += (total_capital - STARTING_CAPITAL) * 0.1

        for j in reversed(range(len(traders))):
            if not traders[j].check_alive():
                ge[j].fitness -= 100
                nets.pop(j)
                ge.pop(j)
                traders.pop(j)
            elif traders[j].credit == STARTING_CAPITAL and traders[j].holdings == 0:
                ge[j].fitness -= 10  # Penalize for not making any trades

    best_trader = max(traders, key=lambda trader: trader.credit + (trader.holdings * current_actual_batch[-1][3]))
    print(f"Best trader's total capital this generation: {best_trader.credit + (best_trader.holdings * current_actual_batch[-1][3])}")
    STARTING_CAPITAL = best_trader.credit
    STARTING_HOLDINGS = best_trader.holdings
    print(f"accumlated total: {STARTING_CAPITAL}")
    print(best_trader.credit_history)
    print(best_trader.holdings_history)

def evaluate_best_genome(best_genome, config):
    trader = Trader(credit=STARTING_CAPITAL, holdings=STARTING_HOLDINGS)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    test_batch = test_batches[0]
    actual_test_batch = actual_test_batches[0]
    
    rows_number = len(test_batch) - 1
    for i in range(10, rows_number):
        if not trader.check_alive():
            break
        decision = net.activate(np.append(test_batch[i], trader.credit / 10000))

        # Apply order
        if decision[0] > 0.9:
            if trader.credit >= decision[1] * 10000:
                trader.buy(decision[1] * 10000, actual_test_batch[i+1][3])
        elif decision[0] < 0.1:
            if trader.holdings >= (decision[1] * 10000) / actual_test_batch[i+1][3]:
                trader.sell(decision[1] * 10000, actual_test_batch[i+1][3])

    final_credit = trader.credit
    final_holdings_value = trader.holdings * actual_test_batch[-1][3]
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

    number_of_generations = 20
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
