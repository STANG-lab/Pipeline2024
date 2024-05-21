import spacy
from collections import Counter


def sep_speakers(path, num_speakers=2):
    # Given a diarized (.srt) file, output dictionary for list of utterances speech for each speaker.
    with open(path, "r", encoding="cp1252") as file:
        lines = file.readlines()
        tags = [f"SPEAKER_0{i}" for i in range(num_speakers)]
        speaker = lambda line: line.split("\t")[1]
        to_txt = lambda line: line.split("\t")[-1].rstrip("\n")
        speakers = {tag: [to_txt(line) for line in lines if speaker(line) == tag] for tag in tags}
    return speakers


class UDFeaturer:

    def __init__(self, lines):
        self.lines = lines
        self.model = spacy.load("en_core_web_sm")
        self.modal_counts = Counter()
        self.sents = self.proc_lines()
        # for sent in self.sents:
        #     print(sent)
        self.max_depths = self.get_max_depth_list()
        self.max_depth = max(self.max_depths)

    def get_max_depth_list(self):
        # multi_tok_utts = [sent for sent in self.sents if type(sent) != spacy.tokens.token.Token]
        return [self.get_max_depth(sent.root) for sent in self.sents]

    def get_max_depth(self, root):
        # Gets depth of rooted, non-binary UD tree.
        if root.n_lefts > 1 or root.n_rights > 1:
            return 1 + max([self.get_max_depth(child) for child in root.children])
        else:
            return 0

    def proc_lines(self):
        sents = []
        for line in self.lines:
            doc = self.model(line)
            self.get_modals(doc)
            sents += list(doc.sents)
        return sents

    def get_modals(self, doc):
        for token in doc:
            # print(token.text.lower())
            modal_auxes = {"can", "could", "may", "might", "must", "ought", "shall", "should", "'ll", "will", "would"}
            if token.pos_ == "AUX" and token.text.lower() in modal_auxes:
                self.modal_counts[token.text.lower()] += 1


def UD_featurize_study(study, out):
    # print(study)
    # print(out)
    for f in study.glob("*"):
        speakers = sep_speakers(f)
        # speakers = sep_speakers("../Studies/Example_study/ASRoutput/example_study_clean/14063_2.srt")
        with open(out, "w") as out_file:
            # print(f.name, file=out_file)
            for key in speakers:
                u = UDFeaturer(speakers[key])
                # print(u.max_depths)
                # print(u.modal_counts)
                row = "\t".join([f.name, key, str(u.max_depths), str(u.modal_counts)])
                print(row, file=out_file)
