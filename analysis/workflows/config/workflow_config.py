import yaml


class WorkflowConfig:
    """
    Attributes:
    -----------
        object_selection:
        event_selection:
        corrections_config:
        histogram_config:
    """

    def __init__(
        self,
        object_selection,
        event_selection,
        corrections_config,
        histogram_config,
        datasets,
        mva=None,
        inference=None,
        combine=None,
        negrw=None,
    ):
        self.object_selection = object_selection
        self.event_selection = event_selection
        self.corrections_config = corrections_config
        self.histogram_config = histogram_config
        self.datasets = datasets
        self.mva = mva
        self.inference = inference
        self.combine = combine
        self.negrw = negrw

    def to_dict(self):
        """Convert WorkflowConfig to a dictionary."""
        d = {
            "object_selection": self.object_selection,
            "event_selection": self.event_selection,
            "corrections_config": self.corrections_config,
            "histogram_config": self.histogram_config.to_dict(),
            "datasets": self.datasets,
        }
        for k in ("mva", "inference", "combine", "negrw"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    def to_yaml(self):
        """Convert WorkflowConfig to a YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)
