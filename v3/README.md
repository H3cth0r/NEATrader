# NEAT Trader Bot

## Changes
- Now using Intraday data.
- batch size = nummber of days * 390
- Batch is not a moving window



## Performance

- **Average Returns**: The bot typically achieves average returns between 30-40% over a period of 2 years. Witht Intraday data, you got to augment the credit or capita to get good returns.
- **Fitness varies**: because this is not an spanning window, each batch can be very different. To the previous one.

## TODO
- Apply moving windows batch, to check if this makes it better.
- Apply predictions and give past data to the current state.
