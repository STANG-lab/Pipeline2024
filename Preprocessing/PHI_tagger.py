import spacy
from pathlib import Path

def clean_line(line):
    # This is applied before PHI tagging so that the [tagging symbols] aren't deleted
    punctuation = '"#$%&()*+,/:;<=>@[\\]^_`{|}~'
    for punct in list(punctuation):
        line = line.replace(punct, "")
    return line

def ent_hlight(doc):
    # Highlights entities with [ entity ] according to the spaCy-detected entity label.
    PHI_labs = {"PERSON", "GPE", "ORG", "EVENT", "DATE"}
    rel_ents = [ent for ent in doc.ents if ent.label_ in PHI_labs]
    starts = set([ent.start for ent in rel_ents])
    ends = set([ent.end for ent in rel_ents])
    i = 0
    highlighted_txt = []
    while i < len(doc):
        if i in starts:
            highlighted_txt.append("[")
        if i in ends:
            highlighted_txt.append("]")
        highlighted_txt.append(doc[i].text + doc[i].whitespace_)
        i += 1
    return "".join(highlighted_txt)

def tag_phi(path, outpath, model):
    # Creates a new file containing the cleaned, PHI flagged data for the file at path.
    with open(outpath, "w") as o, open(path, "r") as f:
        for row in f.readlines():
            text = row.split("\t")[-1].rstrip("\n")
            doc = model(text)
            hl_doc = ent_hlight(doc)
            new_row = "\t".join([*row.split("\t")[:-1], hl_doc])
            print(new_row, file=o)

def tag_study(study_clean, out_dir):
    # out_dir = Path("../Studies/" + study + "/PHIFlagged")
    # out_dir.mkdir(parents=True, exist_ok=True)
    # Creates filtered, PHI flagged files for each .srt file in a given study path
    nlp = spacy.load("en_core_web_sm")
    for file in study_clean.glob("*.tsv"):
        outpath = out_dir / file.name
        tag_phi(file, outpath, nlp)

# if __name__ == "__main__":
#     tag_study("Example_study")
