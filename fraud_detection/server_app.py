"""fraud-detection: A Flower / PyTorch app."""

import torch
import wandb
from typing import Iterable, Optional, Tuple, List
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import matplotlib.pyplot as plt

from fraud_detection.task import Net

app = ServerApp()

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_losses: List[float] = []
        self.round_accuracies: List[float] = [] # <-- Add a list for accuracies

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[float], MetricRecord]:
        """Aggregate evaluation losses and accuracies, then log them."""

        # Get the aggregated loss and metrics from the super class
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, replies)

        # The aggregated accuracy is in the metrics dictionary
        # FedAvg by default names the aggregated accuracy metric 'accuracy'
        aggregated_accuracy = aggregated_metrics.get("accuracy")

        log_dict = {"round": server_round}
        
        if aggregated_loss is not None:
            print(f"Round {server_round}: Aggregated loss = {aggregated_loss}")
            self.round_losses.append(aggregated_loss)
            log_dict["aggregated_loss"] = aggregated_loss

        if aggregated_accuracy is not None:
            print(f"Round {server_round}: Aggregated accuracy = {aggregated_accuracy}")
            self.round_accuracies.append(aggregated_accuracy)
            log_dict["aggregated_accuracy"] = aggregated_accuracy

        # Log both metrics to Weights & Biases in a single step
        wandb.log(log_dict)

        return aggregated_loss, aggregated_metrics
@app.main()
def main(grid:Grid, context:Context):
    wandb.init(
        project="flower-fraud-detection",
        name=f"run-{context.run_id}",
        config=context.run_config,
    )

    fraction_train:float = context.run_config["fraction-train"]
    num_rounds:int = context.run_config["num-server-rounds"]
    lr:float = context.run_config["lr"]

    global_model = Net()

    arrays = ArrayRecord(global_model.state_dict())

    strategy = CustomFedAvg(fraction_train=fraction_train)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr":lr}),
        num_rounds=num_rounds,
    )

    print("saving final model")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pth")
