"""Legacy scalar API backed by the dimension-general benchmark implementation."""
from v16.core import Policy as _Policy, clipped_objective, explained_variance, gae, net


class Policy(_Policy):
    """Preserve the V15 scalar constructor and state-dict layout."""

    def __init__(self, n):
        super().__init__(n, action_dim=1)
