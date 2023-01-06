""" Utility classes to help with distributed training. """
import signal
import time
import uuid
from datetime import datetime

import os
import sys
from typing import Union
from pathlib import Path

import numpy as np
import torch.cuda
import torch.distributed as td
from speedrun.utils.yaml_utils import dump_yaml
from speedrun.distributed.utils import sync_values, gather
from speedrun.distributed.infra_spec import SLURM


class SlurmDistributor(object):
    @property
    def in_distributed_environment(self):
        return SLURM.in_distributed_environment

    @property
    def job_uuid(self):
        uuid = self.get_arg("uuid", os.getenv("SPEEDRUN_UUID"))
        assert uuid is not None
        return uuid

    @property
    def read_config_signal_path(self):
        return os.path.join(self.log_directory, f"read_config_{self.job_uuid}.signal")

    def mark_config_as_ready(self):
        # WARNING: exist_ok should ideally be False, because it's very much not
        # okay if the signal file already exists. But setting it to False leads
        # to mysterious errors when jobs are resumed after preemption (even though
        # they shouldn't), and we don't quite know why this file exist. Until we
        # figure that out, we'll play cowboy and set it to True.
        Path(self.read_config_signal_path).touch(exist_ok=True)
        return self

    def mark_config_as_unready(self):
        Path(self.read_config_signal_path).unlink(missing_ok=True)
        return self

    @property
    def config_is_ready(self):
        return Path(self.read_config_signal_path).exists()

    def wait_for_config_file(
        self, file_name: str = "train_config.yml", read: bool = True
    ):
        # Sleep for a while before querying the config file, just for good measure.
        time.sleep(5)
        while not self.config_is_ready:
            time.sleep(5)
        # Use the extra 5 seconds for the config to have been fully written out,
        # just in case
        time.sleep(5)
        if read:
            # Read it in now
            self.read_config_file(file_name=file_name)

    def dist_console(self, message: str):
        if self.get_arg("verbose", True):
            print(message, file=sys.stderr)
        return self

    def setup_parent_proc(self):
        self.record_args()
        self.auto_setup(
            dump_configuration=self.get_arg("speedrun.dump_configuration", True)
        )
        self.mark_config_as_ready()
        return self

    def setup_child_proc(self):
        self.record_args()
        self.parse_experiment_directory()
        # The parent_proc writes out the config file, which the child_proc reads
        self.wait_for_config_file(read=True)
        return self

    def distributed_setup(self):
        assert self.in_distributed_environment
        # WARNING: Don't read self.is_parent_proc, it doesn't work because
        # is_distributed is not set yet.
        if SLURM.rank == 0:
            self.setup_parent_proc()
        else:
            self.setup_child_proc()
        # WARNING: It's imperative that the line below remain where it is.
        self.set("distributed/is_distributed", True)
        self.init_distributed()
        self.register_signal_handlers()

    def init_distributed(self):
        # Set the environment variables for distributed
        if SLURM.num_nodes == 1:
            # This should be 127.0.0.1, but idk and don't wanna bork anything right now
            os.environ["MASTER_ADDR"] = SLURM.launch_node_ip_address
        else:
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        os.environ["MASTER_PORT"] = str(self.get_arg("distributed_port", 32914))  # noqa

        # Print stuff if we're verbosing
        if self.get_arg("dist_verbose", False):
            self.dist_console("General:")
            # noinspection DuplicatedCode
            self.dist_console(f"  MASTER_ADDR      = {os.environ['MASTER_ADDR']}")
            self.dist_console(f"  MASTER_PORT      = {os.environ['MASTER_ADDR']}")
            self.dist_console(f"  self.rank        = {self.rank}")
            self.dist_console(f"  self.world_size  = {self.world_size}")
            self.dist_console("SlurmSpec:")
            SLURM.print_info(self.dist_console)

        try:
            td.init_process_group(
                backend="nccl",
                world_size=self.world_size,
                rank=self.rank,
            )
            self.dist_console("Successfully initialized distributed.")
            self.dist_console(f"Rank / World Size = {self.rank} / {self.world_size}")
        except RuntimeError:
            # Print debug statements
            self.dist_console("RuntimeError when attempting to init process group.")
            self.dist_console(f"  MASTER_ADDR      = {os.environ['MASTER_ADDR']}")
            self.dist_console(f"  MASTER_PORT      = {os.environ['MASTER_ADDR']}")
            self.dist_console(f"  self.rank        = {self.rank}")
            self.dist_console(f"  SLURM.rank       = {SLURM.rank}")
            self.dist_console(f"  self.world_size  = {self.world_size}")
            self.dist_console(f"  SLURM.world_size = {SLURM.world_size}")
            self.dist_console("Traceback follows.")
            raise
        assert torch.cuda.is_available(), "Something is messed up."
        torch.cuda.set_device(self.device_id)

    @property
    def is_distributed(self):
        return self.get("distributed/is_distributed", False)

    @property
    def world_size(self):
        if self.is_distributed:
            return SLURM.world_size
        else:
            return 1

    @property
    def rank(self):
        if self.is_distributed:
            return SLURM.rank
        else:
            return 0

    @property
    def is_parent_proc(self):
        if self.is_distributed:
            return self.rank == 0
        else:
            return True

    @property
    def device_id(self):
        if self.is_distributed:
            return SLURM.device_id
        else:
            return 0

    @property
    def device(self):
        if self.is_distributed:
            return SLURM.device
        else:
            return self.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def sync(self):
        if not self.is_distributed or not td.is_available():
            return
        td.barrier()

    def sync_values(
        self,
        value: Union[torch.Tensor, np.ndarray, int, float],
        reduction: str = "mean",
    ):
        if not self.is_distributed:
            return value
        return sync_values(
            value=value,
            reduction=reduction,
            device=self.device,
            world_size=self.world_size,
        )

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module
        else:
            return model

    def wrap_model(
        self,
        model: torch.nn.Module,
        find_unused_parameters: bool = False,
        set_static_graph: bool = False,
    ) -> Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]:
        if not self.is_distributed:
            return model
        if len(list(model.parameters())) == 0:
            # We don't wrap if there are no parameters
            return model
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.device_id],
            find_unused_parameters=find_unused_parameters,
        )
        if set_static_graph:
            # noinspection PyProtectedMember
            model._set_static_graph()
        return model

    def get_local_batch_size(self, global_batch_size: int) -> int:
        if not self.is_distributed:
            return global_batch_size
        assert (global_batch_size % self.world_size) == 0
        local_batch_size = global_batch_size // self.world_size
        return local_batch_size

    @property
    def requeue_request_directory(self):
        request_path = os.path.expanduser("~/.salvo_requeue_requests")
        os.makedirs(request_path, exist_ok=True)
        return request_path

    def request_requeue(self, reason: str = None):
        # If this happens, we'll need to write out enough info for
        # the job to be resumed by some daemon.
        if self.is_parent_proc:
            request = {
                "experiment_directory": self.experiment_directory,
                "requeue_request_at": time.time(),
                "request_from_slurm_job_id": SLURM.job_id,
                "reason": reason,
            }
            requeue_request_file_name = (
                f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                f"_{uuid.uuid4().hex}.yml"
            )
            request_file_path = os.path.join(
                self.requeue_request_directory, requeue_request_file_name
            )
            dump_yaml(request, request_file_path)
            return request_file_path

    def preemption_panic(self):
        # If this happens, we'll need to write out enough info for
        # the job to be resumed by some daemon.
        if self.is_parent_proc:
            self.request_requeue(reason="preemption")

    def register_signal_handlers(self):
        if self.is_parent_proc:
            signal.signal(signal.SIGUSR1, self.preemption_panic)


class BigBirdDistributor:
    def squawk(self, message):
        print(f"BigBirdDistributor says: {message}")


DistributorMixin = SlurmDistributor if SLURM.is_available else BigBirdDistributor