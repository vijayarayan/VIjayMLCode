from os import path
import pandas

PROJECT_DIR = (path.abspath(path.join(path.dirname(__file__), "..")))
RESOURCE_DIR = path.abspath(path.join(PROJECT_DIR, "resources"))
OUTPUT_DIR = path.abspath(path.join(PROJECT_DIR, "output"))
DATA_DIR = path.abspath(path.join(PROJECT_DIR, "data"))


def get_absolutepath_data(*file_paths):
    """
    Get the absolute path to some file in the data directory.
    :param file_paths: Path to the file from within the data directory. If your file is
    data/dir1/dir2/test.txt, then call load_resource("dir1", "dir2", "test.txt")
    :return: The absolute path to your requested file
    """
    return path.abspath(path.join(DATA_DIR, *file_paths))
def load_resource(*file_paths):
    """
    Get the absolute path to some file in the resources directory.
    :param file_paths: Path to the file from within the resources directory. If your file is
    resources/dir1/dir2/test.txt, then call load_resource("dir1", "dir2", "test.txt")
    :return: The absolute path to your requested file
    """
    return path.abspath(path.join(RESOURCE_DIR, *file_paths))


_CLUSTER_FILES = {
    "DP": (None),
    "GC": (None),
    "GM": (None),
}

_AFS_FILES = {
    "train": {
        "DP": load_resource("experiment_data/afs/mt1_truelabel_dp_gt55_argQualPairsTop2000_Trainfeatures_norm.csv"),
        "GC": load_resource(
            "experiment_data/afs/mt1_truelabel_gc_gt55_argQual_yes_gt3_comb_sim_max-sent-incl-10_sort_Top2000_Trainfeatures_norm.csv"),
        "GM": load_resource("experiment_data/afs/mt1_truelabel_gm_gt55_argQualPairsTop2000_Trainfeatures_norm.csv"),
    },
    "test": {
        "DP": load_resource("experiment_data/afs/mt1_truelabel_dp_gt55_argQualPairsTop2000_Testfeatures_norm.csv"),
        "GC": load_resource(
            "experiment_data/afs/mt1_truelabel_gc_gt55_argQual_yes_gt3_comb_sim_max-sent-incl-10_sort_Top2000_Testfeatures_norm.csv"),
        "GM": load_resource("experiment_data/afs/mt1_truelabel_gm_gt55_argQualPairsTop2000_Testfeatures_norm.csv"),
    }
}

_FACET_QUALITY_FILES = {
    "train": {
        "DP": load_resource("experiment_data/aq_SigDial2015/facet_quality/dp-slider-means.training.pmi.csv"),
        "EVO": load_resource("experiment_data/aq_SigDial2015/facet_quality/evo-slider-means.training.pmi.csv"),
        "GC": load_resource("experiment_data/aq_SigDial2015/facet_quality/gc-slider-means.training.pmi.csv"),
        "GM": load_resource("experiment_data/aq_SigDial2015/facet_quality/gm-slider-means.training.pmi.csv"),
    },
    "test": {
        "DP": load_resource("experiment_data/aq_SigDial2015/facet_quality/dp-slider-means.testing.pmi.csv"),
        "EVO": load_resource("experiment_data/aq_SigDial2015/facet_quality/evo-slider-means.testing.pmi.csv"),
        "GC": load_resource("experiment_data/aq_SigDial2015/facet_quality/gc-slider-means.testing.pmi.csv"),
        "GM": load_resource("experiment_data/aq_SigDial2015/facet_quality/gm-slider-means.testing.pmi.csv"),
    }
}


def get_output_file(filename):
    """
    Gets the absolute path to a file (which may not exist) in the output directory of the project.
    :param filename: The filename. For example: "test.txt", not "output/test.txt"
    :return: The absolute path to the file
    """
    return path.abspath(path.join(OUTPUT_DIR, filename))


def load_aq_data():
    """
    Loads the data for training/testing the argument quality regressor.
    GoodSliderMean is the true label.
    :return:
    """
    result = dict()
    for subset, files in _FACET_QUALITY_FILES.items():
        if subset not in result:
            result[subset] = dict()
        for topic, file in files.items():
            if topic not in result[subset]:
                result[subset][topic] = dict()
            result[subset][topic] = pandas.read_csv(file)

    return result


def load_afs_feature_data():
    """
    loads the data for training/testing the afs regressor
    :return:
    """
    # not_features = ["Prediction", "sentenceId", "PMI", "datasetId", "bin", "umbc_additional", "_1", "_2",
    #                 "No_Count", "Yes_Count", "discussionId", "postId", "similarity", "uniqueRowNo",
    #                 "sentence", "Id_", "regression_label", "HIT", "count_annotation", "NGramIntersectionSizeNormalized",
    #                 "liwc_dep_overlap_norm", "UMBC_COMBINED_AUG_OVERLAP_NORMALIZED"
    #                 ]
    feature_name_patterns = ["LIWC_", "NGramCosine", "rouge_", "liwc_simplified_dep_overlap_norm", "UmbcSimilarity"]
    meta_information_cols = ["datasetId", "discussionId", "postId", "sentenceId", "sentence"]
    result = dict()
    for subset, files in _AFS_FILES.items():
        if subset not in result:
            result[subset] = dict()
        for topic, file in files.items():
            if topic not in result[subset]:
                result[subset][topic] = dict()
            df = pandas.read_csv(file)
            result[subset][topic]["meta"] = pandas.concat(
                [df[col + "_" + str(i)] for i in range(1, 3) for col in meta_information_cols],
                axis=1, keys=[col + "_" + str(i) for i in range(1, 3) for col in meta_information_cols])

            regression_label = df["regression_label"].values
            result[subset][topic]["true_label"] = regression_label

            feature_cols = [df[col] for col in df.columns if any(col.startswith(pat) for pat in feature_name_patterns)]
            feature_dicts = [dict() for _ in range(len(df))]
            for feature_col in feature_cols:
                assert(len(feature_col.values) == len(feature_dicts))
                for feature_val, feature_dict in zip(feature_col.values, feature_dicts):
                    feature_dict[feature_col.name] = feature_val
                # for feature_val, feature_dict in zip(feature_cols[feature], feature_dicts):
                #     if float(feature_val) != 0.0:
                #         feature_dict[feature] = feature_val

            result[subset][topic]["features"] = feature_dicts

    return result


def load_cluster_feature_data():
    """
    loads the data for clustering
    :return:
    """
    return {
        "GC": pandas.read_csv(_CLUSTER_FILES["GC"]),
        "GM": pandas.read_csv(_CLUSTER_FILES["GM"]),
        "DP": pandas.read_csv(_CLUSTER_FILES["DP"]),
    }


def load_high_quality_instances():
    """
    Loads high quality instances (as determined by yesCount)
    :return:
    """

    def clean_annotated_data(df):
        """
        Cleans MT annotated data
        :param df: DataFrame to clean
        :return: A clean DataFrame
        """
        droppable_columns = [col for col in df.columns.values if col.startswith("Id_") or col.startswith("HIT")]
        return df.drop(droppable_columns, 1)

    return {
        "GC": clean_annotated_data(pandas.read_csv(load_resource("experiment_data/clustering/gc_yesCount_gt3.csv"))),
    }


def get_used_data_lookup():
    """
    :return: a set of all previously used sentences from the AFS and AQ experiments
    """
    seen_set = set()
    for topic, df in load_afs_feature_data().items():
        for col_name in ["sentence_1", "sentence_2"]:
            for row in df[[col_name]].values:
                # rows are single entry, use index 0
                seen_set.add(row[0])
                seen_set.add(row[0].strip())

    for topic, fh in _FACET_QUALITY_FILES.items():
        df = pandas.read_csv(fh)
        for row in df[["Phrase.x"]].values:
            seen_set.add(row[0])
            seen_set.add(row[0].strip())

    return seen_set


if __name__ == "__main__":
    data = load_afs_feature_data()
    print(data["train"]["GC"]["features"][0])
    print(len(data["train"]["GC"]["features"][0]))
