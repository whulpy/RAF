class AgentError(RuntimeError):
    """Base class for errors exposed at the RAF integration boundary."""


class ConfigurationError(AgentError):
    pass





class UnknownSourceError(AgentError):
    pass


class SourceUnavailableError(AgentError):
    pass


class SourceExhaustedError(SourceUnavailableError):
    """A finite source has no remaining records; distinct from a network failure."""


class SourceAuthenticationError(SourceUnavailableError):
    pass


class SampleDownloadError(AgentError):
    pass


class InvalidSampleError(AgentError):
    pass


class DuplicateSampleError(AgentError):
    pass


class MissingRequiredLabelError(AgentError):
    pass


class AcquisitionExhaustedError(AgentError):
    """Raised after bounded, same-source retries cannot produce a candidate."""
