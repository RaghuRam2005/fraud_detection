"""fraud-detection: A Flower / PyTorch app."""

import torch
import wandb
from typing import Iterable, Optional, Tuple
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import matplotlib.pyplot as plt

from fraud_detection.task import Net

app = ServerApp()

class CustomFedAvg(FedAvg):
  def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[float], MetricRecord]:
        """Aggregate evaluation losses and log them."""

        # Get the aggregated loss from the super class
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, replies)

        if aggregated_loss is not None:
            print(f"Round {server_round}: Aggregated evaluation loss = {aggregated_loss}")
            # Store the aggregated loss
            self.round_losses.append(aggregated_loss)
            # Log to Weights & Biases
            wandb.log({"round": server_round, "aggregated_loss": aggregated_loss})

        return aggregated_loss, aggregated_metrics

@app.main()
def main(grid:Grid, context:Context):

  fraction_train:float = context.run_config["fraction-train"]
  num_rounds:int = context.run_config["num-server-rounds"]
  lr:float = context.run_config["lr"]

  global_model = Net()

  arrays = ArrayRecord(global_model.state_dict())

  strategy = FedAvg(fraction_train=fraction_train)

  result = strategy.start(
      grid=grid,
      initial_arrays=arrays,
      train_config=ConfigRecord({"lr":lr}),
      num_rounds=num_rounds,
  )

  print("saving final model")
  state_dict = result.arrays.to_torch_state_dict()
  torch.save(state_dict, "final_model.pth")
