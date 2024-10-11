class SkipGramSampler(BaseSampler):
    __slots__ = '_sample'

    def __init__(self, graph: GraphDataset, seed: int = None):
        super().__init__(graph, seed)
        self._sample = None

    @property
    def targets(self):
        return self._sample[0, :]

    @property
    def contexts(self):
        return self._sample[1, :]

    @property
    def labels(self):
        return self._sample[2, :]

    @property
    def weights(self):
        return self._sample[3, :]

    def _generate_skipgrams(self, sequence, set_size, window_size=4, negative_samples=1.0, shuffle=True,
                            categorical=False, sampling_table=None, seed=None):

        random = self._rs
        if seed is not None:
            random = RandomState(seed)

        couples = []
        labels = []
        for i, wi in enumerate(sequence):
            if not wi:
                continue
            if sampling_table is not None:
                if sampling_table[wi] < random.random():
                    continue

            window_start = max(0, i - window_size)
            window_end = min(len(sequence), i + window_size + 1)
            for j in range(window_start, window_end):
                if j != i:
                    wj = sequence[j]
                    if not wj:
                        continue
                    couples.append([wi, wj])
                    if categorical:
                        labels.append([0, 1])
                    else:
                        labels.append(1)

        if negative_samples > 0:
            num_negative_samples = int(len(labels) * negative_samples)
            words = [c[0] for c in couples]
            random.random.shuffle(words)

            couples += [
                [words[i % len(words)], random.random.randint(1, set_size - 1)]
                for i in range(num_negative_samples)
            ]
            if categorical:
                labels += [[1, 0]] * num_negative_samples
            else:
                labels += [0] * num_negative_samples

        if shuffle:
            if seed is None:
                seed = random.random.randint(0, int(10e6))  # cast to int python3.10 > result in float, and type_error
            random.random.seed(seed)
            random.random.shuffle(couples)
            random.random.seed(seed)
            random.random.shuffle(labels)

        return couples, labels

    def generate(self, data, window_size: int = None, num_neg_samples: int = 1, size: int = None, seed: int = None):
        assert window_size is not None and window_size > 0, "window size parameter must be > 0"
        assert size is not None and size > 0, "size parameter must be > 0"

        ex_weights = defaultdict(int)

        with tqdm(total=len(data), leave=True, desc="Generating Samples.....") as pbar:
            for seq in data:
                pairs, labels = self._generate_skipgrams(seq,
                                                         set_size=size,
                                                         window_size=window_size,
                                                         negative_samples=num_neg_samples)

                for i, p in enumerate(pairs):
                    pair = p
                    label = labels[i]
                    target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
                    if target != context:
                        entry = (target, context, label)
                        ex_weights[entry] += 1
                pbar.update(1)

        sample = np.ndarray((4, len(ex_weights)), dtype=int)
        itm = 0
        for entry, weight in ex_weights.items():
            target, context, label = entry
            sample[0, itm] = target
            sample[1, itm] = context
            sample[2, itm] = label
            sample[3, itm] = weight
            itm += 1

        self._sample = sample

    def sample(self, **kwargs):
        pass

    def loader(self, subset: Subset, **kwargs):
        pass