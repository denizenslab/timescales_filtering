"""Helper functions for saving stimulus features."""
import numpy as np
import os

from hard_coded_things import featuresets_dict
from features import Features
from textgrid import TextGrid


def get_feature(featureset, features_object):
    model_function, model_kwargs = featureset[0]
    modeldata = getattr(features_object, model_function)(**model_kwargs)
    return modeldata


def load_story_info(story_name: str, featureset_name: str = None):
    """Load stimulus info about story."""
    grids = load_grids_for_stories([story_name])
    trfiles = load_generic_trfiles([story_name])
    features_object = Features(grids, trfiles)

    if not featureset_name:
        model_data = None
    else:
        featureset = featuresets_dict[featureset_name]
        featureset[0][1]["downsample"] = False
        model_data = get_feature(featureset, features_object)

    # Get words and word times.
    word_presentation_times = features_object.wordseqs[story_name].data_times
    tr_times = features_object.wordseqs[story_name].tr_times

    if featureset_name:
        story_data = model_data[story_name]
        assert len(story_data) == len(word_presentation_times)
    else:
        story_data = None

    # Get num_words.
    featureset = featuresets_dict["numwords"]
    num_words_feature = get_feature(featureset, features_object)[story_name]
    return story_data, word_presentation_times, tr_times, num_words_feature


def load_grid(story, grid_dir="../data/sentence_TextGrids"):
    """Loads the TextGrid for the given [story] from the directory [grid_dir].
    The first file that starts with [story] will be loaded, so if there are
    multiple versions of a grid for a story, beward.
    """
    filenames = os.listdir(grid_dir)
    gridfilename = [gf for gf in filenames if gf.startswith(story)][0]
    gridfile = TextGrid.load(os.path.join(grid_dir, gridfilename))
    return gridfile


def load_grids_for_stories(stories, grid_dir="../data/sentence_TextGrids"):
    """Loads grids for the given [stories], puts them in a dictionary."""
    print("load_grids_for_stories", stories, grid_dir)
    return dict([(st, load_grid(st, grid_dir)) for st in stories])


class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus presentation code."""
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr

        if trfilename is not None:
            self.load_from_file(trfilename)

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename]."""
        # Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label == "sound-start":
                self.soundstarttime = time

            elif label == "sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))

        # Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes > (itrtimes.mean() * 1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            # Insert new TR where it was missing..
            newtrtime = self.trtimes[btr] + self.expectedtr
            newtrs.append((newtrtime, btr))

        for ntr, btr in newtrs:
            self.trtimes.insert(btr + 1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR."""
        self.trtimes = list(np.arange(ntrs) * self.expectedtr)

    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound."""
        return np.array(self.trtimes) - self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run."""
        return np.diff(self.trtimes).mean()


def load_generic_trfiles(stories, root="../data/trfiles"):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given responses.
    """
    trdict = dict()

    for story in stories:
        trf = TRFile(os.path.join(root, "%s.report" % story))
        trdict[story] = [trf]

    return trdict


def get_mirrored_matrix(original_matrix: np.ndarray, mirror_length: int):
    """Concatenates mirrored versions of the matrix to the beginning and end.
    Used to avoid edge effects when filtering.

    Parameters:
    ----------
    original_matrix : np.ndarray
        num_samples x num_features matrix of original matrix values.
    mirror_length : int
        Length of mirrored segment. If longer than num_samples, num_samples is used as mirror_length.

    Returns:
    -------
    mirrored_matrix : np.ndarray
        (num_samples + 2 * mirror_length) x num_features mirrored matrix.
    """
    mirrored_matrix = np.concatenate(
        [
            original_matrix[:mirror_length][::-1],
            original_matrix,
            original_matrix[-mirror_length:][::-1],
        ],
        axis=0,
    )
    return mirrored_matrix


def get_unmirrored_matrix(
    mirrored_matrix: np.ndarray, mirror_length: int, original_num_samples: int
):
    """Retrieves portion of mirrored matrix corresponding to original matrix.
    Parameters:
    ----------
    mirrored_matrix : np.ndarray
        (original_num_samples + 2 * mirror_length) x num_features mirrored matrix.
    mirror_length : int
        Length of mirrored segment. If longer than num_samples, num_samples is used as mirror_length.
    original_num_sample : int
        Number of samples in original (pre-mirroring) matrix.

    Returns:
    -------
    unmirrored_matrix : np.ndarray
        original_num_samples x num_features unmirrored matrix.
    """
    return mirrored_matrix[mirror_length : mirror_length + original_num_samples]
