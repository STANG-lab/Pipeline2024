from pathlib import Path


class FolderStructure:
    """
    This is an object that creates and encapsulates a directory structure
    Folder paths can be accessed from FolderStructure(out_path, args)[DIR_NAME/SUBDIR_NAME]
    """
    def __init__(self, out_path, args):
        self.out_path = out_path
        self.poss_fnames = ["Raw",
                      "Raw/denoised",
                      "ASRoutput",
                      "ASRoutput/raw",
                      "ASRoutput/clean",
                      "SplitTranscripts",
                      "Features",
                      "Features/sentence_features",
                      "Features/word_aggregates",
                      "Features/4_word_features",
                      "SAD",
                      "SAD/prelab",
                      "SAD/postlab",
                      "SAD/logs",
                      "SAD/megasegs",
                      "Denoised",
                      "PHIFlagged"
                      "openSMILE",
                      "openSMILE/Output",
                      "openSMILE/LabOutput",
                      "openSMILE/logs"]  # Adding to this list adds a folder to the default tree.
        self.dir_tree = self.create_dir_tree(args)

    def create_dir_tree(self, args):
        # Creates dir tree in out_path from list of subdirs
        out_dirs = {}
        if args.full_tree:
            make_new = {dname: True for dname in self.poss_fnames}
        else:
            make_new = self.args_to_dir_tree(args)
        for subdir in make_new:
            new_dir = Path(self.out_path) / subdir
            if make_new[subdir]:
                new_dir.mkdir(parents=True, exist_ok=True)
            out_dirs[subdir] = new_dir
        return out_dirs

    def args_to_dir_tree(self, args):
        # This function makes the appropriate dir tree given args.
        # full tree:
        # returns out_dirs dict and makes directories at the same time.
        fnames = ["Raw"]  # There must always be a folder named "Raw" with the audio data inside.
        if args.asr:
            fnames.append("ASRoutput")
            fnames.append("ASRoutput/raw")
            fnames.append("ASRoutput/clean")
        if args.snr or args.UD or args.take_denoised:  # ADD FEATURES HERE TO CREATE DIRS FOR THEM AUTOMATICALLY
            fnames.append("Features")
        if args.word_agg:
            fnames.append("Features/word_aggregates")
        if args.sentence_features:
            fnames.append("Features/sentence_features")
        # if args.prep_annotations:
        #     fnames.append("SplitTranscripts")
        if args.time:
            fnames.append("SAD")
            fnames.append("SAD/prelab")
            fnames.append("SAD/postlab")
            fnames.append("SAD/megasegs")
            fnames.append("SAD/logs")
        if args.acoustics:
            fnames.append("openSMILE")
            fnames.append("openSMILE/Output")
            fnames.append("openSMILE/LabOutput")
            fnames.append("openSMILE/logs")
        if args.word_features:
            fnames.append("Features/4_word_features")
        if args.phi_flag:
            fnames.append("PHIFlagged")
        if args.take_denoised:
            fnames.append("Denoised")
            fnames.append("Raw/denoised")
        if args.snr:
            fnames.append("Raw/denoised")
        make_new = {}
        for dir_name in fnames:
            make_new[dir_name] = True
        for name in [el for el in self.poss_fnames if el not in make_new.keys()]:
            make_new[name] = False
            # Elements of make_new are True if that directory needs to be created.
        return make_new

    def list_files(self, dir_name, extensions):
        # Returns a list of audio files in dir_name.
        return [p.resolve() for p in self[dir_name].glob("**/*") if p.suffix in extensions]


    def same_name(self, dir1, dir2):
        pass

    def __getitem__(self, item):
        # Allows direct access
        return self.dir_tree[item]
