import os
import time
from typing import Any, Union, Iterable
from contextlib import contextmanager

try:
    import persistqueue
except ImportError:
    persistqueue = None


class QueueMixin(object):
    @property
    def queues(self) -> dict:
        if not hasattr(self, "_queues"):
            setattr(self, "_queues", dict())
        return getattr(self, "_queues")

    @property
    def queue_directory(self):
        queue_directory = os.path.join(self.experiment_directory, "Queues")
        if not os.path.exists(queue_directory):
            os.makedirs(queue_directory, exist_ok=True)
        return queue_directory

    def initialize_queue(
        self,
        name: str = "default",
        path: str = None,
        is_ack_queue: bool = False,
        unique_entries: bool = False,
        is_filo: bool = False,
        overwrite: str = False,
        auto_commit: bool = True,
        serializer: str = None,
        **persistqueue_kwargs,
    ):
        assert persistqueue is not None, "Please `pip install persistqueue` first."
        queues = self.queues
        if name in queues and not overwrite:
            # Queue is already added and we don't overwrite it
            return queues[name]
        # Create a path if required
        if path is None:
            path = os.path.join(self.queue_directory, f"{name}")
        # Build queue kwargs
        queue_kwargs = {"path": path}
        if auto_commit is not None:
            queue_kwargs.update({"auto_commit": auto_commit})
        if serializer is not None:
            if isinstance(serializer, str):
                serializer = getattr(persistqueue.serializers, serializer)
            queue_kwargs.update({"serializer": serializer})
        queue_kwargs.update(persistqueue_kwargs)
        # Parse queue class
        if is_ack_queue:
            if unique_entries:
                QueueClass = persistqueue.UniqueAckQ
            elif is_filo:
                QueueClass = persistqueue.sqlackqueue.FILOSQLiteAckQueue
            else:
                QueueClass = persistqueue.SQLiteAckQueue
        else:
            if unique_entries:
                QueueClass = persistqueue.UniqueQ
            elif is_filo:
                QueueClass = persistqueue.FILOSQLiteQueue
            else:
                QueueClass = persistqueue.FIFOSQLiteQueue
        # Make the queue
        new_queue = QueueClass(**queue_kwargs)
        # Register the queue and return
        queues.update({name: new_queue})
        return new_queue

    def is_queue_initialized(self, name: str = "default"):
        return name in self.queues

    def put_in_queue(
        self, value: Any, name: str = "default", **queue_constructor_kwargs
    ):
        if name not in self.queues:
            queue = self.initialize_queue(name=name, **queue_constructor_kwargs)
        else:
            queue = self.queues[name]
        queue.put(value)
        return self

    def get_from_queue(
        self,
        name: str = "default",
        default: Any = None,
        block: bool = False,
        timeout: int = None,
    ):
        if name not in self.queues:
            return default
        queue = self.queues[name]
        try:
            item = queue.get(block=block, timeout=timeout)
        except persistqueue.exceptions.Empty:
            item = default
        return item

    def commit(self, name: str = "default"):
        if name in self.queues:
            self.queues[name].task_done()
        return self

    @contextmanager
    def hold_commits(self, name: Union[str, Iterable[str]] = "default"):
        yield
        if isinstance(name, str):
            self.commit(name)
        elif isinstance(name, (list, tuple)):
            for _name in name:
                self.commit(_name)
        else:
            raise TypeError

    def ack(self, item: Any, name: str = "default"):
        if name in self.queues:
            queue = self.queues[name]
            if hasattr(queue, "ack"):
                queue.ack(item)
        return self

    def nack(self, item: Any, name: str = "default"):
        if name in self.queues:
            queue = self.queues[name]
            if hasattr(queue, "nack"):
                queue.nack(item)
        return self

    def ack_failed(self, item: Any, name: str = "default"):
        if name in self.queues:
            queue = self.queues[name]
            if hasattr(queue, "ack_failed"):
                queue.ack_failed(item)
        return self

    def ack_all(self, items: Iterable[Any], name: str = "default"):
        for item in items:
            self.ack(item=item, name=name)
        return self

    def nack_all(self, items: Iterable[Any], name: str = "default"):
        for item in items:
            self.nack(item=item, name=name)
        return self

    def ack_failed_all(self, items: Iterable[Any], name: str = "default"):
        for item in items:
            self.ack_failed(item=item, name=name)
        return self

    def send_report(
        self,
        name: Union[str, Iterable[str]] = "default",
        include_stats: bool = True,
        **contents,
    ):
        if include_stats:
            stats = {
                "step": self.step,
                "epoch": self.epoch,
                "experiment_directory": self.experiment_directory,
                "time": time.time(),
            }
            contents = {**stats, **contents}
        # This implements multiplexing.
        names = name if isinstance(name, (list, tuple)) else [name]
        for _name in names:
            self.put_in_queue(value=contents, name=_name)
        return self

    def receive_report(
        self,
        name: str = "default",
        ack: bool = False,
        block: bool = False,
        timeout: int = None,
        default: Any = None,
    ):
        class Nothing(object):
            pass

        item = self.get_from_queue(
            name=name, block=block, timeout=timeout, default=Nothing
        )
        if ack and item is not Nothing:
            self.ack(item=item, name=name)
        return item if item is not Nothing else default

    def receive_all_reports(
        self, name: str = "default", ack: bool = False,
    ):
        class Nothing(object):
            pass

        items = []
        while True:
            item = self.receive_report(name=name, ack=ack, block=False, default=Nothing)
            if item is Nothing:
                break
            else:
                items.append(item)
        return items
