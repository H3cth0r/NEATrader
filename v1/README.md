# NEAT Trader Bot

This project implements a trading bot using NeuroEvolution of Augmenting Topologies (NEAT) and a simple feedforward neural network. The bot is designed to trade stocks based on historical data and maximize its capital over time. The bot makes buy and sell decisions based on input features derived from stock data and its current capital.

## Trader Class

The `Trader` class simulates a trading agent with the following attributes and methods:

- **Attributes**:
  - `credit`: The current capital available for trading.
  - `holdings`: The current amount of stock holdings.
  - `current_state`: The current state represented by an array of features.
  - `is_alive`: A boolean indicating if the trader is still active.
  - `goal`: A boolean indicating if the trader has achieved its goal (not used in this version).
  - `time_spent`: The amount of time the trader has been active (not used in this version).
  - `credit_history`: A list tracking the history of the trader's credit.
  - `holdings_history`: A list tracking the history of the trader's holdings.

- **Methods**:
  - `check_alive()`: Checks if the trader is still active based on its credit.
  - `buy(quantity, price)`: Executes a buy order.
  - `sell(quantity, price)`: Executes a sell order.
  - `update_state(new_state)`: Updates the current state of the trader.

## Neural Network and Training

The neural network is configured with 15 inputs and 2 outputs (buy/sell decision and amount). It uses sigmoid activation functions and has one hidden layer. The training process involves the following steps:

1. **Initialization**: Initialize traders, neural networks, and genomes.
2. **Decision Making**: For each data point, the neural network makes a buy/sell decision.
3. **Order Execution**: Execute the buy/sell orders based on the network's decision.
4. **Fitness Evaluation**: Evaluate the fitness of each trader based on their profit/loss and apply penalties for invalid actions.
5. **Survival**: Remove traders that run out of credit.
6. **Best Trader**: Identify and print the best trader's performance for each generation.

## Performance

- **Average Returns**: The bot typically achieves average returns between 30-40% over a period of 2 years.
- **Best Performance**: Occasionally, the bot achieves returns over 50% in the same period.
- **Learning Curve**: The bot may take some time to learn and start growing the capital. Performance can vary significantly between runs.

## Configuration

- **Transaction Fee**: A transaction fee of 0.1% is applied to discourage excessive trading.
- **Dataset**: Historical stock data for Microsoft (MSFT) from 2022 is used.
- **NEAT Configuration**: The NEAT algorithm configuration is defined in `config-feedforward`.
