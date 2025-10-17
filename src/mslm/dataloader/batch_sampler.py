from torch.utils.data import Sampler
from transformers.trainer_pt_utils import get_length_grouped_indices

class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, mega_batch_mult: int = None, generator=None):
        lengths = [len(item) for item in dataset]
        self.indices = get_length_grouped_indices(
            lengths=lengths,
            batch_size=batch_size,
            mega_batch_mult=mega_batch_mult,
            generator=generator,
        )

        self.batches = [
            self.indices[i : i + batch_size]
            for i in range(0, len(self.indices), batch_size)
        ]

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)