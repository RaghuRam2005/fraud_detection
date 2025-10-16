"""fraud-detection: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fraud_detection.task import Net, load_data
from fraud_detection.task import test as test_fn
from fraud_detection.task import train as train_fn

app = ClientApp()

@app.train()
def train(msg:Message, context:Context):
  model = Net()
  model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  partition_id = context.node_config["partition-id"]
  num_partitions = context.node_config["num-partitions"]
  trainloader, _ = load_data(partition_id, num_partitions)

  train_loss = train_fn(
      model,
      trainloader,
      epochs=context.run_config["local-epochs"],
      lr=msg.content["config"]["lr"],
      device=device,
  )

  model_record = ArrayRecord(model.state_dict())
  metrics = {
      "train-loss":train_loss[-1],
      "loss-history":train_loss,
      "num-examples":len(trainloader.dataset),
  }
  metrics_record = MetricRecord(metrics)
  content = RecordDict({"arrays":model_record, "metrics":metrics_record})
  return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg:Message, context:Context):
  model = Net()
  model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  partition_id = context.node_config["partition-id"]
  num_partitions = context.node_config["num-partitions"]
  _, valloader = load_data(partition_id, num_partitions)

  eval_loss, eval_acc = test_fn(
      model,
      valloader,
      device=device,
  )

  metrics = {
      "eval-loss" : eval_loss,
      "eval-acc" : eval_acc,
      "num-examples" : len(valloader.dataset),
  }

  metric_record = MetricRecord(metrics)
  content = RecordDict({"metrics":metric_record})
  return Message(content=content, reply_to=msg)
