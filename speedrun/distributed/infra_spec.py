import os
from typing import Callable
from abc import ABC, abstractmethod
from speedrun.distributed.utils import sync_values, gather

import torch.cuda
import torch.distributed as dist


class AbstractClusterSpec(ABC):
    @property
    @abstractmethod
    def distributed_is_initialized(self):
        pass

    @property
    @abstractmethod
    def in_distributed_environment(self):
        pass

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def world_size(self):
        pass

    @property
    @abstractmethod
    def node_id(self):
        pass

    @property
    @abstractmethod
    def num_nodes(self):
        pass

    @property
    @abstractmethod
    def local_id(self):
        pass

    @property
    @abstractmethod
    def local_world_size(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def launch_node_ip_address(self):
        pass


class GeneralClusterSpec(AbstractClusterSpec):
    @property
    def distributed_is_initialized(self):
        return dist.is_available() and dist.is_initialized()

    @property
    def in_distributed_environment(self):
        if self.distributed_is_initialized:
            return True
        else:
            return self.check_externally_if_in_distributed_environment()

    def check_externally_if_in_distributed_environment(self):
        raise NotImplementedError

    @property
    def rank(self):
        if self.distributed_is_initialized:
            return dist.get_rank()
        elif not self.in_distributed_environment:
            return 0
        else:
            return self.get_rank_externally()

    def get_rank_externally(self):
        raise NotImplementedError

    @property
    def world_size(self):
        if self.distributed_is_initialized:
            return dist.get_world_size()
        elif not self.in_distributed_environment:
            return 1
        else:
            return self.get_world_size_externally()

    def get_world_size_externally(self):
        raise NotImplementedError

    @property
    def node_id(self):
        if not self.in_distributed_environment:
            return 0
        else:
            return self.get_node_id_externally()

    def get_node_id_externally(self):
        raise NotImplementedError

    @property
    def num_nodes(self):
        if not self.in_distributed_environment:
            return 1
        else:
            return self.get_num_nodes_externally()

    def get_num_nodes_externally(self):
        raise NotImplementedError

    @property
    def local_id(self):
        return self.get_local_id_externally()

    def get_local_id_externally(self):
        raise NotImplementedError

    @property
    def local_world_size(self):
        try:
            return self.get_local_world_size_externally()
        except NotImplementedError:
            return

    def get_local_world_size_externally(self):
        raise NotImplementedError

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_id)
        else:
            return torch.device("cpu")

    @property
    def device_id(self):
        return self.local_id

    @property
    def launch_node_ip_address(self):
        return self.get_launch_node_ip_address_externally()

    def get_launch_node_ip_address_externally(self):
        raise NotImplementedError

    @property
    def job_id(self):
        return self.get_job_id_externally()

    def get_job_id_externally(self):
        raise NotImplementedError

    def print_info(self, printer: Callable[[str], None]):
        if printer is None:
            printer = print
        attries = [
            "rank",
            "world_size",
            "num_nodes",
            "local_id",
            "device",
            "device_id",
            "launch_node_ip_address",
            "job_id",
        ]
        for attr in attries:
            try:
                attr_value = getattr(self, attr)
            except NotImplementedError:
                attr_value = "Not implemented"
            message = f"{self.__class__.__name__}.{attr} = {attr_value}"
            printer(message)
        return self


class SlurmSpec(GeneralClusterSpec):
    def is_available(self):
        return os.getenv("SLURM_JOB_ID") is not None

    def check_externally_if_in_distributed_environment(self):
        return int(os.getenv("SLURM_NTASKS", 1)) > 1

    def get_world_size_externally(self):
        return int(os.getenv("SLURM_NTASKS", 1))

    def get_node_id_externally(self):
        return int(os.getenv("SLURM_NODEID", 0))

    def get_num_nodes_externally(self):
        return int(os.getenv("SLURM_NNODES", 1))

    def get_launch_node_ip_address_externally(self):
        return os.getenv("SLURM_LAUNCH_NODE_IPADDR")

    def get_local_id_externally(self):
        return int(os.getenv("SLURM_LOCALID", 0))

    def get_job_id_externally(self):
        return os.getenv("SLURM_JOB_ID")

    def get_rank_externally(self):
        return os.getenv("SLURM_PROCID")


def detect_cluster_and_get_cluster_spec():
    # TODO
    pass


SLURM = SlurmSpec()


class _LegacySlurmSpec(object):
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
            "device_id",
        ]
        for attry in attries:
            message = f"SlurmSpec.{attry} = {getattr(self, attry, 'UNKNOWN')}"
            printer(message)
        return self
