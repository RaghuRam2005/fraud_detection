"""fraud-detection: A Flower / PyTorch app."""

import torch
from typing import Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import matplotlib.pyplot as plt

from fraud_detection.task import Net

app = ServerApp()

def plot_loss_curve(loss_list):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-', linewidth=2)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


class CustomFedAvg(FedAvg):
  def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""

        for reply in replies:
            if reply.has_content():
                # Retrieve the ConfigRecord from the message
                train_losses = reply.content["loss-history"]
                plot_loss_curve(train_losses)
                
        # Aggregate the ArrayRecords and MetricRecords as usual
        return super().aggregate_train(server_round, replies)

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
