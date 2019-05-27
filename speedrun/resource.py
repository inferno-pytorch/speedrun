import os
import time


class WaiterMixin(object):
    """
    Class to wait on a process to finish before starting the computation.
    """
    @staticmethod
    def is_alive(pid):
        """Checks if a process with PID `pid` is alive."""
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    def wait_for_pid(self, pid, timeout=1):
        """Wait till the process with PID `pid` is dead."""
        while True:
            if not self.is_alive(pid):
                break
            else:
                time.sleep(timeout)
                continue

    # noinspection PyUnresolvedReferences
    def wait(self):
        """
        This function blocks until a specified process has terminated. This can be useful for
        pipelining multiple experiments. You may call this function anytime after `auto_setup` is
        called, e.g. in the `__init__` of your experiment or before the training training loop.

        The commandline arguments this function listens to are:
            `--wait.for`: specifies the PID of the process to wait for.
            `--wait.check_interval`: Interval to query the status of the process being waited for.
            `--wait.verbose`: Whether to print info.

        Example
        -------
        This is assuming that your file calls this function somewhere.

        $ python my_script.py TEST-0 --wait.for 1234 --wait.check_interval 10 --wait.verbose True

        This will wait for the process with PID 1234. While doing so, it will check its status
        every 10 seconds.

        Warning
        -------
            May destroy friendships.
        """

        pid_to_wait_for = self.get_arg('wait.for', None)
        timeout = self.get_arg('wait.check_interval', 1)
        verbose = self.get_arg('wait.verbose', True)
        if pid_to_wait_for is None:
            return
        if verbose:
            message = f"Waiting for PID {pid_to_wait_for} to finish (my PID is {os.getpid()})..."
            (self.print if hasattr(self, 'print') else print)(message)
        self.wait_for_pid(pid_to_wait_for, timeout)
        if verbose:
            message = f"Done waiting for PID {pid_to_wait_for}. It's showtime!"
            (self.print if hasattr(self, 'print') else print)(message)
        return True
