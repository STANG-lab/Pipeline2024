# Call all other processes, meant just to hold args and defaults, and to control imports judiciously when possible.



import argparse
import subprocess
from cleanASR import clean
from make_eafs import get_eafs
from folder_holder import FolderStructure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=str)

    # General utility
    parser.add_argument("--out_path", help="Optional out path, defaults to args.study.", default=None)
    parser.add_argument("--full_tree", help="Automatically builds full directory structure.", default=False)
    # parser.add_argument("--prep_annotations", help="Splits transcripts into speaker by speaker .tsvs.", default=False)
    parser.add_argument('--take_denoised', help="Use denoised data for further acoustic processing. Performs SNR.", default=False)

    # Alex features, note that acoustic features require completed transcripts, as well as time and snr.
    parser.add_argument('--snr', help="Measure SNR and record features", default=False)
    parser.add_argument('--time', help="Get timing, turn, and other assorted SAD features", default=False)
    parser.add_argument('--acoustics', help="Get acoustic features with SMILE (Requires transcripts)", default=False)

    # Preprocessing commands
    parser.add_argument('--asr', help="Process files with ASR", default=True)
    parser.add_argument('--override_asr', help="Redo all transcriptions for study", default=False)  # This should usually stay False.
    parser.add_argument('--phi-flag', help="PHI flag data", default=False)
    parser.add_argument('--word_agg', help="Generate word aggregates from data (3)", default=False)

    # Feature generation commands.
    parser.add_argument('--UD', help="Generate UD features", default=False)
    parser.add_argument('--word_features', help="Generate lexical features (4)", default=False)

    # TODO: Encapsulate these in Allen NLP environment
    parser.add_argument('--sentence_features', help="Generate sentence features", default=False)
    parser.add_argument('--coref', help="Generate coreference discourse features", default=False)
    parser.add_argument('--semgraph', help="Generate semantic graph features", default=False)

    args = parser.parse_args()

    if args.out_path is not None:
        out_path = args.out_path
        # Custom out path creation
    else:
        out_path = f"../Studies/{args.study}"
        # TODO: Make this more flexible

    out_dirs = FolderStructure(out_path, args)
    asr_clean = out_dirs["ASRoutput/clean"]  # Naming this because it will be used a couple of times.
    snr = args.snr
    if args.take_denoised:
        # If denoising is desired, SNR needs to be performed
        snr = True
    target_audios = out_dirs["Raw"]
    if snr:
        from Acoustic_000_SignalNoiseRatio import *
        SNRMeasurer(target_audios, out_dirs["Features"], out_dirs['Raw/denoised']).run_SNR_pipe()
        # if not args.take_denoised:
        #     subprocess.run(["rm", "-r", str(out_dirs["Raw/denoised"])])
        # if not args.snr:  # This check is why the above logic resets the value of the intermediate "snr" variable.
        #     # In case Denoised data is required, but it's unnecessary to keep the SNR data, it is deleted.
        #     subprocess.run(["rm", f"{out_dirs['Features']}/snr_final.csv"])

    if args.take_denoised:
        # Copies Raw data to Denoised, then copies denoised data on top of it, replacing identical filenames.
        # Acoustic features require Denoised samples to be separated
        raw_files = out_dirs.list_files('Raw', {".mp4", ".wav", ".mp3", ".m4a"})
        dn_files = out_dirs.list_files('Raw/denoised', {".mp4", ".wav", ".mp3", ".m4a"})
        # raw_files = [p.resolve() for p in out_dirs['Raw'].glob("**/*") if p.suffix in {".mp4", ".wav", ".mp3", ".m4a"}]
        # dn_files = [p.resolve() for p in out_dirs['Raw/denoised'].glob("**/*") if p.suffix in {".mp4", ".wav", ".mp3", ".m4a"}]
        for fl in raw_files + dn_files:
            subprocess.run(["cp", fl, str(out_dirs['Denoised'])])
        target_audios = out_dirs["Denoised"]
        # out_dirs.set_target_audios("Denoised")

    if args.asr:
        auds = (p.resolve() for p in target_audios.glob("**/*") if p.suffix in {".mp4", ".wav", ".mp3", ".m4a"})
        for targ_aud in auds:
            # With this check, data can be added to a study without reapplying ASR to old files.
            if not args.override_asr:
                existing_trans = set([e.name for e in out_dirs.list_files("ASRoutput/raw", {".srt"})])
                # existing_trans = set([e.name for e in out_dirs["ASRoutput/raw"].glob("*")])
                same_name = lambda x, y: x.split(".")[0] == y.split(".")[0]
                if any([same_name(targ_aud.name, transcript) for transcript in existing_trans]):
                    continue
            # My goal is to abstract this check and apply it for every step.
            subprocess.run(["whisperx", targ_aud,
                            "--model", "medium",
                            "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
                            "--batch_size", "1",
                            "--min_speakers", "1",
                            "--max_speakers", "2",
                            "--highlight_words", "False",
                            "--language", "en",
                            "--diarize",
                            "--hf_token", "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"])
            subprocess.run(f"mv *.srt {out_dirs['ASRoutput/raw']}/", shell=True)
            # Make sure to add / after out_dirs[...] when using mv in subprocess.

            # Whisper produces a bunch of files for each audio, and unfortunately I can't turn it off.
            # To avoid space deadlock, after each file is processed the auxiliary files are deleted.
            # Just MAKE SURE none of this filetype that you want to keep go into the same folder as this script!!!!!
            subprocess.run("rm *.vtt", shell=True)
            subprocess.run("rm *.json", shell=True)
            subprocess.run("rm *.tsv", shell=True)
            subprocess.run("rm *.txt", shell=True)  # The shell option allows for pattern matching with *.
        # Do not parallelize:  GPU parallelization does not play nicely with CPU encapsulation.
        # clean(out_dirs["ASRoutput/raw"], asr_clean)

    text_feature_target = asr_clean
    if args.phi_flag:
        from PHI_tagger import tag_study
        tag_study(text_feature_target, out_dirs["PHIFlagged"])
        # text_feature_target = out_dirs["PHIFlagged"] # Set feature generation to be from PHIflagged data for testing.
    # if args.prep_annotations:
    #     get_eafs(out_dirs["PHIFlagged"], out_dirs["SplitTranscripts"])
    if args.UD:
        from UDFeatures import UD_featurize_study
        UD_featurize_study(text_feature_target, out_dirs["Features"] / "UDFeatures.tsv")
    word_agg = args.word_agg or args.word_features
    if word_agg:
        from Transcript_003_WordAggregates import get_word_aggregates
        get_word_aggregates(text_feature_target, out_dirs["Features"], out_dirs["Features/word_aggregates"], name="3_wordaggregates.csv")
    if args.word_features:
        from Transcript_004_WordFeatures import get_word_features
        get_word_features(out_dirs, text_feature_target)
    if args.time:
        from time_extraction import get_time_features
        get_time_features(target_audios, text_feature_target, out_dirs)

    # # TODO: ISOLATE THESE WITH ASSERT STATEMENT
    # if args.sentence_features:  # install allennlp and DialogTag
    #     from Transcript_005_SentenceFeatures import get_sent_feats
    #     get_sent_feats(out_dirs)






