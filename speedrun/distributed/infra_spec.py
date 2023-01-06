import os
from typing import Callable
import torch.cuda
from utils import sync_values, gather


class SlurmSpec(object):

    @property
    def is_available(self):
        return self.in_distributed_environment

    @property
    def job_id(self):
        return os.getenv("SLURM_JOB_ID")

    @property
    def num_tasks(self):
        return int(os.getenv("SLURM_NTASKS", 1))

    @property
    def tasks_per_node(self):
        return self.num_tasks // self.num_nodes

    @property
    def local_id(self):
        return int(os.getenv("SLURM_LOCALID", 0))

    @property
    def num_nodes(self):
        return int(os.getenv("SLURM_NNODES", 1))

    @property
    def node_id(self):
        return int(os.getenv("SLURM_NODEID", 0))

    @property
    def rank(self):
        return (self.node_id * self.tasks_per_node) + self.local_id

    @property
    def world_size(self):
        return self.num_tasks

    @property
    def launch_node_ip_address(self):
        return os.getenv("SLURM_LAUNCH_NODE_IPADDR")

    @property
    def in_distributed_environment(self):
        return self.num_tasks > 1

    @property
    def device_id(self):
        return self.local_id

    @property
    def device(self):
        if torch.cuda.is_available():
            return f"cuda:{self.device_id}"
        else:
            return "cpu"

    def gather(self, tensor: torch.Tensor, preserve_gradients: bool = False):
        if self.in_distributed_environment:
            return gather(
                tensor,
                world_size=self.world_size,
                rank=self.rank,
                preserve_gradients=preserve_gradients,
            )
        else:
            # We simulate the output of gather.
            if preserve_gradients:
                return [tensor.clone()]
            else:
                return [torch.empty_like(tensor).copy_(tensor.data)]

    def sync_values(
        self,
        value,
        reduction: str,
        preserve_gradients: bool = False,
        raise_when_not_in_distributed_environment: bool = False,
    ):
        if self.in_distributed_environment:
            return sync_values(
                value=value,
                reduction=reduction,
                device=self.device,
                world_size=self.world_size,
                rank=self.rank,
                preserve_gradients=preserve_gradients,
            )
        else:
            if raise_when_not_in_distributed_environment:
                raise RuntimeError("Can't sync when not in distributed env.")
            else:
                return value

    def print_info(self, printer: Callable = None):
        if printer is None:
            printer = print
        attries = [
            "job_id",
            "num_tasks",
            "tasks_per_node",
            "local_id",
            "num_nodes",
            "node_id",
            "rank",
            "world_size",
            "launch_node_ip_address",
            "in_distributed_environment",
            "device_id"
        ]
        for attry in attries:
            message = f"SlurmSpec.{attry} = {getattr(self, attry, 'UNKNOWN')}"
            printer(message)
        return self

SLURM = SlurmSpec()


