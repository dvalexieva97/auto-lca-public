from datetime import datetime
from logging import log, INFO, warning
import time
import csv
import os


def log_time(log_file_path_getter=None):
    """
    A decorator that logs the execution time and other metadata like rowid and version.

    :param log_file_path_getter: A callable that returns the log file path and accepts (self, rowid, version)
    """
    default_logger_folder = "src/auto_lca/output/logs/"

    def decorator(func):
        def wrapper(self, *args, **kwargs):

            header = ["id", "pipeline_step", "start_time", "end_time", "exec_time"]
            start_time = datetime.now()

            result = func(self, *args, **kwargs)

            end_time = datetime.now()
            exec_time = end_time - start_time

            rowid = args[0]
            # TODO: if not log_file_path_getter:
            log_file_path = log_file_path_getter()
            add_header = False
            if not os.path.exists(log_file_path):
                add_header = True

            print(add_header, "header")

            with open(log_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                if add_header:
                    writer.writerow(header)

                writer.writerow(
                    [
                        rowid,
                        func.__name__,
                        start_time.isoformat(),
                        end_time.isoformat(),
                        exec_time.total_seconds(),
                    ]
                )

            return result

        return wrapper

    return decorator


# TODO
class Logger:
    def __init__(self) -> None:
        pass

    def log(self, msg, level=None):
        if not level:
            level = INFO
        return log(msg)


loger = Logger()

# loger.log(msg="blabla")
