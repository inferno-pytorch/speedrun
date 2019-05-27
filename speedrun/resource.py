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
            if self.is_alive(pid):
                break
            else:
                time.sleep(timeout)
                continue

    # noinspection PyUnresolvedReferences
    def wait(self):
        """Wait for a process to finish before starting the current."""
        pid_to_wait_for = self.get_arg('--wait.for', None)
        timeout = self.get_arg('--wait.check_interval', 1)
        verbose = self.get('--wait.verbose', False)
        if verbose:
            message = f"Waiting for PID {pid_to_wait_for} to finish (my PID is {os.getpid()})..."
            (self.print if hasattr(self, 'print') else print)(message)
        self.wait_for_pid(pid_to_wait_for, timeout)
        if verbose:
            message = f"Done waiting for PID {pid_to_wait_for}. It's showtime!"
            (self.print if hasattr(self, 'print') else print)(message)
        return True
