"""
tasks to kick-off the project
"""

from invoke import task


@task
def formatter(tsk, fix=False):
    """
    python format
    """
    auto_fix = "" if fix else "--check --diff"
    cmd = " && ".join(
        [
            f"python -m black src {auto_fix}",
        ]
    )
    tsk.run(cmd, echo=True, pty=True)


@task
def lint(tsk):
    """
    python lint
    """
    cmd = " && ".join(
        [
            "python -m flake8 *.py app/ tests/ --statistics",
        ]
    )
    tsk.run(cmd, echo=True, pty=True)
