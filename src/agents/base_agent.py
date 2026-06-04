# src/agents/base_agent.py
# Abstract base class for all agents.


class BaseAgent:
    """Minimal interface shared by all agents."""

    def act(self, observation):
        """Return an action given an observation vector."""
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """Run a training step or episode."""
        raise NotImplementedError
