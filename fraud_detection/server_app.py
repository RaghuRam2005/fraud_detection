"""fraud-detection: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fraud_detection.task import Net

app = ServerApp()

@app.main()
def main(grid:Grid, context:Context):

    fraction_train:float = context.run_config["fraction-train"]
    num_rounds:int = context.run_config["num-server-rounds"]
    lr:float = context.run_config["lr"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_train=fraction_train,
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
