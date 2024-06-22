import numpy as np
import pandas as pd
from statisco import StockDataFrame
from trader import Trader
import neat

from functionalities import plot_live_data

# Constants
STARTING_CAPITAL = 1000
STARTING_HOLDINGS = 0
N_DAYS_BATCH = 5
TEST_PERCENTAGE = 0.4
CONFIG_FILE = "./config-feedforward"

# Load Data
DATA = pd.read_csv("../resources/WSM_intraday_1min.csv")
DATA['Date'] = pd.to_datetime(DATA['Date'])
DATA["Adj Close"] = DATA["Close"]
DATA.set_index('Date', inplace=True)
DATA = DATA.sort_values(by="Date")
DATA = StockDataFrame(data=DATA)

# Apply calculations to the dataset
DATA.calculate(close_returns=True)
DATA.calculate(sma=True, interval=3)
DATA.calculate(ema=True, interval=3, smooth=3)
DATA.calculate(wma=True, interval=3)
DATA.calculate(atr=True, interval=3)
DATA.calculate_MACD(short_window=12, long_window=26, signal_window=9)
COLUMN_INDEX_DICT = {column_name: index for index, column_name in enumerate(DATA.columns)}
Data_NOT_N = DATA.copy().to_numpy()

# Calculate available number of rows per day
ROWS_PER_DAY = DATA.resample("D").size().to_numpy()
ROWS_PER_DAY = ROWS_PER_DAY[ROWS_PER_DAY!=0]

# Normalize Data
DATA = DATA.normalize(inplace=False)

# Create batches based on days
def create_batches(data, batches_sizes):
    pre_i = 0
    batches = []
    for batch_s in batches_sizes:
        batches.append(data[pre_i:pre_i+batch_s])
        pre_i += batch_s
    return batches
btchs = np.array([ROWS_PER_DAY[btch:btch+4].sum() for btch in range(0, len(ROWS_PER_DAY), 4)])
calc_batch_range = lambda test_size, btch_list: (int((1-test_size) * len(btch_list)),int(-(test_size * len(btch_list))))
train_btch_range, test_btch_range = calc_batch_range(TEST_PERCENTAGE, btchs)

train_batch_size = btchs[:test_btch_range]
test_batch_size = btchs[test_btch_range:]

# Create batches
train_batches = create_batches(DATA[:train_batch_size.sum()], train_batch_size)
test_batches = create_batches(DATA[train_batch_size.sum():], test_batch_size)
not_n_train_batches = create_batches(Data_NOT_N[:train_batch_size.sum()], train_batch_size)
not_n_test_batches = create_batches(Data_NOT_N[train_batch_size.sum():], test_batch_size)

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
    current_actual_batch = not_n_train_batches[batch_counter]
    # batch_counter += 1
    batch_counter = (batch_counter + 1) % len(train_batches)
    
    rows_number = len(current_normalized_batch) - 1
    for i in range(10, rows_number):
        for j in range(len(traders)):
            if not traders[j].check_alive():
                continue
            decision = nets[j].activate(np.append(current_normalized_batch[i], traders[j].credit / 10000))

            # Apply order
            if decision[0] > 0.6:
                if traders[j].credit >= decision[1] * 10000:
                    traders[j].buy(decision[1] * 10000, current_actual_batch[i+1][3])
                else:
                    ge[j].fitness -= 10  # Penalize for trying to buy without enough credit
            elif decision[0] < 0.4:
                if traders[j].holdings >= (decision[1] * 10000) / current_actual_batch[i+1][3]:
                    traders[j].sell(decision[1] * 10000, current_actual_batch[i+1][3])
                else:
                    ge[j].fitness -= 10  # Penalize for trying to sell without enough holdings

            # Apply fitness function
            total_capital = traders[j].credit + (traders[j].holdings * current_actual_batch[i+1][3])
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
    # STARTING_CAPITAL = best_trader.credit
    # STARTING_HOLDINGS = best_trader.holdings
    print(f"accumlated total: {STARTING_CAPITAL}")
    print(best_trader.credit_history)
    print(best_trader.holdings_history)

def evaluate_best_genome(best_genome, config):
    STARTING_CAPITAL = 1000
    STARTING_HOLDINGS = 0
    trader = Trader(credit=STARTING_CAPITAL, holdings=STARTING_HOLDINGS)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    for test_batch, not_n_test_batch in zip(test_batches, not_n_test_batches):
        rows_number = len(test_batch) - 1
        for i in range(rows_number):
            if not trader.check_alive():
                break
            decision = net.activate(np.append(test_batch[i], trader.credit / 10000))

            # Apply order
            if decision[0] > 0.6:
                print(f"buy: {decision[1]}")
                if trader.credit >= decision[1] * 10000:
                    trader.buy(decision[1] * 10000, not_n_test_batch[i+1][3])
            elif decision[0] < 0.4:
                print(f"sell: {decision[1]}")
                if trader.holdings >= (decision[1] * 10000) / not_n_test_batch[i+1][3]:
                    print("BBBBBB")
                    trader.sell(decision[1] * 10000, not_n_test_batch[i+1][3])

        STARTING_CAPITAL = trader.credit
        STARTING_HOLDINGS = trader.holdings
        final_credit = trader.credit
        final_holdings_value = trader.holdings * not_n_test_batch[-1][3]
        total_capital = final_credit + final_holdings_value
        print(f"Total Capital: {total_capital}")
        print(f"Final Credit: {final_credit}")
        print(f"Final Holdings Value: {final_holdings_value}")
        print("Credit History:", trader.credit_history)
        print("Holdings History:", trader.holdings_history)

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

    number_of_generations = 10
    winner = population.run(training, number_of_generations)

    print(f"\nBest genome: \n{winner}")

    # Evaluate the best genome
    evaluate_best_genome(winner, config)

def main():
    run(CONFIG_FILE)
    pass

if __name__ == "__main__":
    main()
