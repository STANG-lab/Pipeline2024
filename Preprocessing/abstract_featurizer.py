

class Featurizer:
    """
    This is an abstract object that holds information from a FolderHolder object
    This object standardizes folder names across the pipeline.
    When new components are created, they can be integrated in such a way that functions fall in the scope of an object which inherits Featurizer.
    That way they have access to the folder structure through the names here.
    All of these objects would be taking outdirs as an argument and naming some components anyway.
    """
    def __init__(self, outdirs):
        self.outdirs = outdirs
        self.word_agg = outdirs["Features/word_aggregates"]
        self.sent_feats = outdirs["Features/5_sentence_features"]
        self.feat_dir = outdirs["Features"]

        self.disc_feats = outdirs["Features/6_discourse_features"]
        self.str_graph_dir = outdirs["Features/6_discourse_features/6b_struct_graphs"]

        self.sad_prelab = outdirs["SAD/prelab"]
        self.sad_postlab = outdirs["SAD/postlab"]
        self.sad_megasegs = outdirs["SAD/megasegs"]
        self.sad_logs = outdirs["SAD/logs"]


        self.smile_logs = outdirs['openSMILE/logs']
        # log_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/log_files"


        # output_path="/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_std_spellchecked_batch_1+2/"
        # Path(output_path).mkdir(parents=True, exist_ok=True)
        # Path(log_path).mkdir(parents=True, exist_ok=True)

        self.audio_df_path = outdirs["Features"] / "snr_final.csv"


        # audio_path = '/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task'

        # denoised_path = outdirs["Raw/denoised"]
        # denoised_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/data/remora_test/audio_by_task/denoised"

        self.lab_files_path = outdirs["SAD/postlab"]
        # lab_files_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/sad_out/test/post_labs"

        # The folders below are populated by this object, as opposed to the previous which are pre-supposed.
        self.smile_out = outdirs["openSMILE/Output"]
        # output_path = "/Volumes/Alex_R_Music_ssD/Research/ac_pipe/output/smile_out/remora_test/openSMILE_output"
        self.smile_labeled_out = outdirs["openSMILE/LabOutput"]
