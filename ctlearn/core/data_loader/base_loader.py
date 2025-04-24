from abc import ABC, abstractmethod

class BaseDLDataLoader(ABC):

    def __init__(
        self,
        DLDataReader,
        indices,
        tasks,
        batch_size=64,
        random_seed=None,
        sort_by_intensity=False,
        stack_telescope_images=False,
        **kwargs,
    ):

        super().__init__(**kwargs)
        "Initialization"
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.batch_size = batch_size
        self.random_seed = random_seed
        # self.on_epoch_end()
        self.stack_telescope_images = stack_telescope_images
        self.sort_by_intensity = sort_by_intensity

        # Set the input shape based on the mode of the DLDataReader
        if self.DLDataReader.__class__.__name__ != "DLFeatureVectorReader":
            if self.DLDataReader.mode == "mono":
                self.input_shape = self.DLDataReader.input_shape
            elif self.DLDataReader.mode == "stereo":
                self.input_shape = self.DLDataReader.input_shape[
                    list(self.DLDataReader.selected_telescopes)[0]
                ]
                # Reshape inputs into proper dimensions
                # for the stereo analysis with stacked images
                if self.stack_telescope_images:
                    self.input_shape = (
                        self.input_shape[1],
                        self.input_shape[2],
                        self.input_shape[0] * self.input_shape[3],
                    )
     
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass
