"""fraud-detection: A Flower / PyTorch app."""

import torch
import wandb
from typing import Iterable, Optional, Tuple
from flwr.common.typing import MetricRecord
from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fraud_detection.task import Net

app = ServerApp()


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy to log metrics to Weights & Biases."""

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate model updates and log client-side training loss."""
        for reply in replies:
            if reply.has_content() and "metrics" in reply.content:
                node_id = reply.metadata.node_id
                train_loss = reply.content["metrics"]["train-loss"]
                print(f"Round {server_round} | Client {node_id} | Train Loss: {train_loss:.4f}")
                wandb.log({
                    f"train_loss_client_{node_id}": train_loss,
                    "round": server_round
                })
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation results and log the aggregated metrics."""

        aggregated_results = super().aggregate_evaluate(server_round, replies)

        if aggregated_results is not None:
            aggregated_loss = aggregated_results["eval-loss"]
            aggregated_accuracy = aggregated_results["eval-acc"]

            log_dict = {
                "round": server_round,
                "aggregated_eval_loss": aggregated_loss,
                "aggregated_eval_accuracy": aggregated_accuracy,
            }

            print(f"Round {server_round} | Aggregated Eval Loss: {aggregated_loss:.4f} | Aggregated Eval Accuracy: {aggregated_accuracy:.4f}")
            wandb.log(log_dict)

        return aggregated_results


@app.main()
def main(grid:Grid, context:Context):
    """Main function to run the federated learning experiment."""
    wandb.init(
        project="flower-fraud-detection",
        name=f"run-{context.run_id}",
        config=context.run_config,
        define_metric="round/*",
    )

    fraction_train:float = context.run_config["fraction-train"]
    num_rounds:int = context.run_config["num-server-rounds"]
    lr:float = context.run_config["lr"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = CustomFedAvg(
        fraction_train=fraction_train,
        fit_metrics_aggregation_fn=lambda metrics: metrics[0][1],
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr":lr}),
        num_rounds=num_rounds,
    )

    print("Saving final model...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pth")

    wandb.finish()
