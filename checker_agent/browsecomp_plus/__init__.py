"""BrowseComp-Plus trajectory adapters for StepGap-Agent."""

from .adapter import convert_official_record
from .schema import BrowseObservation, BrowseStep, BrowseTrajectory

__all__ = ["BrowseObservation", "BrowseStep", "BrowseTrajectory", "convert_official_record"]
