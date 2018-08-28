import os


class TSVGenerator(object):
    def __init__(self, root_dir='data'):
        self.root_dir = root_dir
        self.phase = ['train', 'dev', 'test']
        self.corpus_path = os.path.join(root_dir, "{}.sen")
        self.label_path = os.path.join(root_dir, "{}.lab")

    def __call__(self, phase):
        assert phase in self.phase, 'Unable phase'

        corpus_path = self.corpus_path.format(phase)
        label_path = self.label_path.format(phase)

        corpus = [line.replace('\n', '').strip()
                  for line in open(corpus_path, 'r').readlines()]
        label = [line.replace('\n', '').strip()
                 for line in open(label_path, 'r').readlines()]

        with open(os.path.join(self.root_dir, f'{phase}.tsv'), 'w') as f:
            for sen, lab in zip(corpus, label):
                f.write('{}\t{}\n'.format(sen, lab))


if __name__ == '__main__':
    generator = TSVGenerator()
    target = ['train', 'dev', 'test']
    for val in target:
        generator(val)
