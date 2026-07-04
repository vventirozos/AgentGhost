"""Ghost Agent evaluation harness.

Local, offline, deterministic outcome eval. Exists so every prompt tweak,
tool change, or swarm retopology becomes falsifiable against a frozen
baseline instead of eyeballed.

The harness runs fully offline by design — Ghost is privacy-by-default
and the eval's job is to measure the agent *as deployed*, i.e. talking
to its own local upstream, never to external services.

Public API:
  - EvalSuite         : runs a list of tasks, collects TaskResults
  - TaskResult        : per-task outcome (pass, steps, tokens, duration)
  - SuiteResult       : aggregate across a suite run
  - load_default_suite: template+regression+curated default suite
  - freeze_baseline   : dump a SuiteResult as the current-snapshot baseline
  - compare_to_baseline: diff a SuiteResult vs a frozen baseline
  - no_external_network: context manager that blocks non-loopback egress
"""

from .metrics import TaskResult, SuiteResult, aggregate
from .tasks import (
    EvalTask,
    ChallengeTemplateTask,
    CuratedRequestTask,
    RegressionProbeTask,
    load_default_suite,
    load_offline_suite,
    load_capability_suite,
    load_post_learning_suite,
)
from .suite import EvalSuite, RunnerCallable
from .baseline import (
    freeze_baseline,
    load_baseline,
    load_baseline_provenance,
    compare_to_baseline,
    baseline_trust_warnings,
)
from .network_guard import no_external_network, NetworkEgressError

__all__ = [
    "TaskResult",
    "SuiteResult",
    "aggregate",
    "EvalTask",
    "ChallengeTemplateTask",
    "CuratedRequestTask",
    "RegressionProbeTask",
    "load_default_suite",
    "load_offline_suite",
    "load_capability_suite",
    "load_post_learning_suite",
    "EvalSuite",
    "RunnerCallable",
    "freeze_baseline",
    "load_baseline",
    "load_baseline_provenance",
    "compare_to_baseline",
    "baseline_trust_warnings",
    "no_external_network",
    "NetworkEgressError",
]
