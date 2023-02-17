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
from torch._C._distributed_c10d import TCPStore

from speedrun.utils.yaml_utils import dump_yaml
from speedrun.distributed.utils import sync_values, gather
from speedrun.distributed.infra_spec import SLURM


def disable_print_for_non_master_processes(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


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

    import time
    import numpy as np
    from typing import Callable, Protocol
    from abc import abstractmethod
    from datetime import timedelta
    from pathlib import Path
    from typing import List, Optional, Iterable, Union
    from speedrun.distributed.utils import sync_values
    import os
    import torch
    import sys
    import torch.distributed as dist
    from torch.distributed.elastic.multiprocessing.errors import record
    from torch._C._distributed_c10d import TCPStore

    class AbstractClusterSpec(Protocol):
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
            # print(f"GeneralClusterSpec.in_distributed_environment: {self.check_externally_if_in_distributed_environment()}")
            if self.distributed_is_initialized:
                return True
            else:
                return self.check_externally_if_in_distributed_environment()

        def check_externally_if_in_distributed_environment(self):
            raise NotImplementedError

        @property
        def rank(self):
            if self.distributed_is_initialized:
                return int(dist.get_rank())
            elif not self.in_distributed_environment:
                return 0
            else:
                return int(self.get_rank_externally())

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

    class LocalSpec(GeneralClusterSpec):
        def check_externally_if_in_distributed_environment(self):
            return int(os.getenv("WORLD_SIZE", 1)) > 1

        def get_world_size_externally(self):
            return int(os.getenv("WORLD_SIZE", 1))

        def get_node_id_externally(self):
            return int(os.getenv("GROUP_RANK", 0))

        def get_num_nodes_externally(self):
            return int(os.getenv("NUM_NODES", 1))

        def get_launch_node_ip_address_externally(self):
            return os.getenv("MASTER_ADDR")

        def get_local_id_externally(self):
            return int(os.getenv("LOCAL_RANK", 0))

        def get_job_id_externally(self):
            return os.getenv("TORCHELASTIC_RUN_ID")

        def get_rank_externally(self):
            return os.getenv("RANK")

    class LocalClusterDistributor(object):
        LOCAL_SPEC = LocalSpec()

        @property
        def in_distributed_environment(self):
            return self.LOCAL_SPEC.in_distributed_environment

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
            try:
                if self.get_arg("verbose", True):
                    print(message, file=sys.stderr, flush=True)
            except:
                print(message, file=sys.stderr, flush=True)
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
        print("This process has rank:", SLURM.rank)
        if int(SLURM.rank) == 0:
            self.setup_parent_proc()
        else:
            self.setup_child_proc()
        # WARNING: It's imperative that the line below remain where it is.
        self.set("distributed/is_distributed", True)
        self.init_distributed()
        self.register_signal_handlers()


    def _init_pg_with_tcp_based_kv(self):
        # In case of the Mila cluster there is not adequate write protection for files when
        # access by multiple nodes. This results deadlocks at process group initialization as soon
        # as the process group is created due to infinite read-loops.
        # This can be avoided by using a TCP-store instead, which does not have this problem.
        store = TCPStore(host_name=os.environ['MASTER_ADDR'],
                         port=int(os.environ['MASTER_PORT']),
                         world_size=self.world_size, is_master=self.rank == 0)
        try:
            td.init_process_group(
                backend="nccl",
                world_size=self.world_size,
                rank=self.rank,
                store=store
            )
            self.dist_console("Successfully initialized distributed.")
            self.dist_console(f"Process Rank {self.rank} using GPU {self.device_id} with World Size {self.world_size}")
        except RuntimeError:
            # Print debug statements
            self.dist_console("RuntimeError when attempting to init process group.")
            self.dist_console(f"  MASTER_ADDR      = {os.environ['MASTER_ADDR']}")
            self.dist_console(f"  MASTER_PORT      = {os.environ['MASTER_PORT']}")
            self.dist_console(f"  self.rank        = {self.rank}")
            self.dist_console(f"  SLURM.rank       = {SLURM.rank}")
            self.dist_console(f"  self.world_size  = {self.world_size}")
            self.dist_console(f"  SLURM.world_size = {SLURM.world_size}")
            self.dist_console("Traceback follows.")
            raise

    def _init_pg_with_file_based_kv(self):
        # this works on CC clusters
        init_method = self.get_arg("dist_url", "env://")
        try:
            td.init_process_group(
                backend="nccl",
                world_size=self.world_size,
                rank=self.rank,
                init_method=init_method
            )
            self.dist_console("Successfully initialized distributed.")
            self.dist_console(f"Process Rank {self.rank} using GPU {self.device_id}) with World Size {self.world_size}")
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

    def init_distributed(self):
        # Set the environment variables for distributed
        if SLURM.num_nodes == 1:
            # This should be 127.0.0.1, but idk and don't wanna bork anything right now
            os.environ["MASTER_ADDR"] = SLURM.launch_node_ip_address
        else:
            if "HOSTNAME" in os.environ:
                os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
            else:
                # the first note in the slurm nodelist is the host adress, can be obtained
                # with this command:
                # $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
                os.system('export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)')
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

        if self.get_arg("backend", "nccl") == "nccl":
            self._init_pg_with_file_based_kv()
        # TCP port is necessary on the Mila cluster, otherwise you run into the risk
        # of erratic deadlocks. This is NOT a problem on CC clusters afaik
        elif self.get_arg("backend", "") == "tcp":
            self._init_pg_with_tcp_based_kv()
        else:
            raise NotImplementedError(f"Unknown backend {self.get_arg('backend', '')}")
        # waiting for all nodes to do their thing
        torch.distributed.barrier()
        assert torch.cuda.is_available(), "Something is messed up."
        torch.cuda.set_device(self.device_id)

        #FIXME: For debugging commented out, but when real training starts with a lot of
        # GPUS you want to avoid redundant log bloat.
        # disable_print_for_non_master_processes(self.rank == 0)


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
        model.to(f"cuda:{self.device_id}" if str(self.device_id).isnumeric() else self.device_id)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[f"cuda:{self.device_id}" if str(self.device_id).isnumeric() else self.device_id],
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