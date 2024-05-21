from collections import defaultdict


def sep_speech(path):
    # Given a diarized (.tsv) file, output list of utterances speech for each speaker.
    with open(path, "r", encoding="utf8") as file:
        lines = file.readlines()
        # tags = [f"SPEAKER_0{i}" for i in range(num_speakers)]
        speaker_text = defaultdict(list)
        for line in lines:
            speaker_text[speaker(line)].append(line.rstrip("\n"))
    return speaker_text


def get_eafs(files, out_path):
    for f in files.glob("*"):
        speaker_text = sep_speech(f)
        for key in speaker_text:
            out = out_path / f"{key}_{f.name.split('.')[0]+'.tsv'}"
            with open(out, "w") as ofile:
                for line in speaker_text[key]:
                    print(line, file=ofile)

# def get_eaf(file, out_path):
#     with open(file, "r") as infile:
#         speaker = lambda l: l.split("\t")[1]
#         unique_speakers = {}
#         for line in file:
#             if speaker(line)
#             out = out_path / f"{key}_{f.name.split('.')[0]+'.tsv'}"
#             with open(out, "w") as ofile:
#                 for line in speaker_text[key]:
#                         print(line, file=ofile)