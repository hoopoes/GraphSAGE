# Components of Generator
# 1) RandomWalk, UniformRandomWalk
# 2) UnsupervisedSampler
# 3) LinkSequence, OnDemandLinkSequence

# LinkSequence, OnDemandLinkSequence: Generator 정의를 위해 필요함

import collections
from collections.abc import Iterable
import numpy as np
from tensorflow.keras.utils import Sequence

from sampler import UnsupervisedSampler
from utils import is_real_iterable, random_state


# Sequence
# Base object for fitting to a sequence of data, such as a dataset.

class LinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit/evaluate/predict`.

    This class generates data samples for link inference models
    and should be created using the :meth: `flow` method of HinSAGELinkGenerator.

    Args:
        sample_function (Callable): A function that returns features for supplied head nodes.
        batch_size
        ids (iterable): Link IDs to batch, each link id being a tuple of (source, destination) node ids.
        targets (list, optional): A list of targets or labels to be used in the downstream task.
        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.
        seed (int, optional): Random seed
    """

    def __init__(
        self, sample_function, batch_size, ids, targets=None, shuffle=True, seed=None
    ):
        # Check that ids is an iterable
        if not is_real_iterable(ids):
            raise TypeError("IDs must be an iterable or numpy array of graph node IDs")

        # Check targets is iterable & has the correct length
        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError("Targets must be None or an iterable or numpy array ")
            if len(ids) != len(targets):
                raise ValueError(
                    "The length of the targets must be the same as the length of the ids")
            self.targets = np.asanyarray(targets)
        else:
            self.targets = None

        # Ensure number of labels matches number of ids
        if targets is not None and len(ids) != len(targets):
            raise ValueError("Length of link ids must match length of link targets")

        # Store the generator to draw samples from graph
        if isinstance(sample_function, collections.abc.Callable):
            self._sample_features = sample_function
        else:
            raise TypeError(
                "({}) The sampling function expects a callable function.".format(
                    type(self).__name__))

        self.batch_size = batch_size
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle
        self._rs, _ = random_state(seed)

        # Shuffle the IDs to begin
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num):
        """
        Generate one batch of data
        Args:
            batch_num (int): number of a batch
        Returns:
            batch_feats (list): Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets (list): Targets/labels for the batch.
        """
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # The ID indices for this batch
        batch_indices = self.indices[start_idx:end_idx]

        # Get head (root) nodes for links
        head_ids = [self.ids[ii] for ii in batch_indices]

        # Get targets for nodes
        batch_targets = None if self.targets is None else self.targets[batch_indices]

        # Get node features for batch of link ids
        batch_feats = self._sample_features(head_ids, batch_num)

        return batch_feats, batch_targets

    def on_epoch_end(self):
        """
        Shuffle all link IDs at the end of each epoch
        """
        self.indices = list(range(self.data_size))
        if self.shuffle:
            self._rs.shuffle(self.indices)


class OnDemandLinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods :meth:`keras.Model.fit`,
    :meth:`keras.Model.evaluate`, and :meth:`keras.Model.predict`

    This class generates data samples for link inference models
    and should be created using the :meth:`flow` method of
    :class:`.GraphSAGELinkGenerator` or :class:`.Attri2VecLinkGenerator`.

    Args:
        sample_function (Callable): A function that returns features for supplied head nodes.
        sampler (UnsupersizedSampler):  An object that encapsulates the neighbourhood sampling of a graph.
            The generator method of this class returns a batch of positive and negative samples on demand.
    """

    def __init__(self, sample_function, batch_size, walker, shuffle=True):
        # Store the generator to draw samples from graph
        if isinstance(sample_function, collections.abc.Callable):
            self._sample_features = sample_function
        else:
            raise TypeError(
                "({}) The sampling function expects a callable function.".format(
                    type(self).__name__
                )
            )

        if not isinstance(walker, UnsupervisedSampler):
            raise TypeError(
                "({}) UnsupervisedSampler is required.".format(type(self).__name__)
            )

        self.batch_size = batch_size
        self.walker = walker
        self.shuffle = shuffle
        # FIXME(#681): all batches are created at once, so this is no longer "on demand"
        self._batches = self._create_batches()
        self.length = len(self._batches)
        self.data_size = sum(len(batch[0]) for batch in self._batches)

    def __getitem__(self, batch_num):
        """
        Generate one batch of data.

        Args:
            batch_num<int>: number of a batch

        Returns:
            batch_feats<list>: Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets<list>: Targets/labels for the batch.

        """

        if batch_num >= self.__len__():
            raise IndexError(
                "Mapper: batch_num larger than number of esstaimted  batches for this epoch."
            )
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # Get head nodes and labels
        head_ids, batch_targets = self._batches[batch_num]

        # Obtain features for head ids
        batch_feats = self._sample_features(head_ids, batch_num)

        return batch_feats, batch_targets

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.length

    def _create_batches(self):
        return self.walker.run(self.batch_size)

    def on_epoch_end(self):
        """
        Shuffle all link IDs at the end of each epoch
        """
        if self.shuffle:
            self._batches = self._create_batches()


