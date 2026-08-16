"""Shared pytest fixtures across all test suites."""
import pytest
import numpy as np

from drift_detector.embedding import DeterministicProvider, LocalTransformerProvider
from drift_detector.baseline import BaselineStore, Baseline


@pytest.fixture
def provider():
    """Fast deterministic test provider for mathematical invariant and unit testing."""
    return DeterministicProvider(dim=32)


@pytest.fixture
def sample_baseline(provider):
    """3-sample distributed systems baseline."""
    texts = [
        "Distributed database replication and consensus protocols like Raft and Paxos.",
        "High-performance caching mechanisms, memory indexing, and horizontal sharding.",
        "Leader election algorithms, Byzantine fault tolerance, and log replication.",
    ]
    return BaselineStore(provider).build(texts)
