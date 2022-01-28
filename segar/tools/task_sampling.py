
from typing import Type

from segar.mdps import Task, Initialization


def spawn_tasks(task_cls: Type[Task], init_cls: Type[Initialization],
                task_args: dict, init_args: dict,
                n_samples: int = 10) -> list[Task]:
    """Generates a set of tasks, given task and initialization types and args.

    :param task_cls: Task class.
    :param init_cls: Initialization class.
    :param task_args: Task arguments.
    :param init_args: Initialization arguments.
    :param n_samples: Number of tasks to generate.
    :return: List of tasks.
    """

    tasks = []
    for n in range(n_samples):
        init = init_cls(**init_args)
        task = task_cls(initialization=init, **task_args)
        tasks.append(task)

    return tasks
