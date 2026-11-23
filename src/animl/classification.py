'''
Tools for Saving, Loading, and Using Species Classifiers

@ Kyra Swanson 2023
'''
import json
from typing import Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
import onnxruntime

from animl import generator, file_management
from animl.utils.general import get_onnx_device, softmax


def load_classifier(model_path: str,
                    classes: Union[int, str, Path, pd.DataFrame] = None,
                    device: Optional[str] = None,
                    quiet: bool = True):
    '''
    Creates a model instance and loads the latest model state weights.

    Args:
        model_path (str): file or directory path to model weights
        classes (int | str | Path | pd.DataFrame): number of classes, path to associated class list, or pd.DataFrame of class list
        device (str): specify to run on cpu or gpu
        architecture (str): expected model architecture

    Returns:
        model: model object of given architecture with loaded weights
        class_list: list of class names
    '''
    class_list = None
    model_path = Path(model_path)
    if not quiet:
        print(f'Loading model at {model_path}')
    # check to make sure GPU is available if chosen
    providers = get_onnx_device(user_set=device, quiet=quiet)

    
    model = onnxruntime.InferenceSession(model_path, providers=providers)
    model.model_type = "classifier"

    # get number of classes
    if isinstance(classes, str) or isinstance(classes, Path):
        class_list = load_class_list(classes)
    elif isinstance(classes, pd.DataFrame):
        class_list = classes
    else:
        class_list = None
    # try to load class dict from metadata
    props = model.get_modelmeta().custom_metadata_map
    if "class_dict" in props:
        if not quiet:
            print("Loaded class_dict from ONNX metadata.")
        class_dict = json.loads(props["class_dict"])
        class_list = [class_dict[str(i)] for i in range(len(class_dict))]
        class_list = pd.DataFrame({'class': class_list})

        # no need to return epoch
    return model, class_list


def load_class_list(classlist_file):
    """
    Return classlist file as pd.DataFrame.

    Args:
        classlist_file (str): file path to class list

    Returns:
        pd.DataFrame of class list
    """
    if not Path(classlist_file).is_file():
        raise FileNotFoundError(f"Class list file not found at {classlist_file}")
    return pd.read_csv(classlist_file)


def classify(model,
             detections,
             resize_width,
             resize_height,
             file_col: str = 'filepath',
             crop: bool = True,
             normalize: bool = True,
             letterbox: bool = False,
             batch_size: int = 1,
             prefetch_size: int = 2,
             num_workers: int = 2,
             out_file: Optional[str] = None,
             quiet: bool = True):
    """
    TODO: align with R version
    Predict species using classifier model.

    Args:
        model: preloaded classifier model
        detections (mult): dataframe of (animal) detections, list of filepaths or filepath str
        resize_width (int): image width input size
        resize_height (int): image height input size
        file_col (str): column name containing file paths
        crop (bool): use bbox to crop images before feeding into model
        normalize (bool): normalize the tensor before inference
        out_file (str): path to save prediction results to; the files that failed to load
            are saved next to it, in classification_failed_files.txt

    Returns:
        tuple of
            np.ndarray: softmaxed predictions, one row per detection, in order, leaving out
                every detection from the files in failed_files
            list: files that failed to load, as in animl-py
    """
    if file_management.check_file(out_file, output_type="Classification results"):
        return file_management.load_data(out_file).to_numpy(), _load_failed_files(out_file)

    # Manifest
    if isinstance(detections, pd.DataFrame):
        if file_col not in detections.columns:
            raise ValueError(f"file_col {file_col} not found in manifest columns")
        # no frame column, assume all images and set to 0
        if 'frame' not in detections.columns:
            print("Warning: 'frame' column not found in manifest columns. Defaulting to 0 assuming images.")
            detections['frame'] = 0

        dataset = generator.ManifestGenerator(manifest=detections,
                                              resize_height=resize_height,
                                              resize_width=resize_width,
                                              file_col=file_col,
                                              crop=crop,
                                              normalize=normalize,
                                              letterbox=letterbox,
                                              batch_size=batch_size,
                                              prefetch_size=prefetch_size,
                                              num_workers=num_workers,  # 4 threads for loading
                                              quiet=quiet,
                                              return_indices=True)

    # Single File
    elif isinstance(detections, str):
        detections = pd.DataFrame({file_col: detections, 'frame': 0}, index=[0])
        dataset = generator.ManifestGenerator(manifest=detections,
                                              resize_height=resize_height,
                                              resize_width=resize_width,
                                              file_col=file_col,
                                              crop=crop,
                                              normalize=normalize,
                                              letterbox=letterbox,
                                              batch_size=batch_size,
                                              prefetch_size=prefetch_size,
                                              num_workers=num_workers,
                                              quiet=quiet,
                                              return_indices=True)
    # List of Files
    elif isinstance(detections, list):
        detections = pd.DataFrame({file_col: detections, 'frame': 0}, index=range(len(detections)))
        dataset = generator.ManifestGenerator(manifest=detections,
                                              resize_height=resize_height,
                                              resize_width=resize_width,
                                              file_col=file_col,
                                              crop=crop,
                                              normalize=normalize,
                                              letterbox=letterbox,
                                              batch_size=batch_size,
                                              prefetch_size=prefetch_size,
                                              num_workers=num_workers,
                                              quiet=quiet,
                                              return_indices=True)
    else:
        raise AssertionError("Input must be a data frame of crops, single file path or vector of file paths.")

    # Predict
    start_time = time()
    # one row per detection, NaN where the image couldn't be loaded;
    # softmax over each image's classes
    by_row = generator.predict_by_row(model, dataset, transform=softmax)
    raw_output, failed_files = _split_failed(by_row, dataset.manifest[file_col])

    if out_file:
        file_management.save_data(pd.DataFrame(raw_output), out_file)
        _save_failed_files(out_file, failed_files)

    processed = len(raw_output)
    print(f"\nFinished classification. Total images processed: {processed} at {round(processed/(time() - start_time), 1)} img/s.")

    return raw_output, failed_files


FAILED_FILES_NAME = "classification_failed_files.txt"


def _split_failed(by_row: np.ndarray, filepaths: pd.Series):
    '''
    Turn predict_by_row output into animl-py's (predictions, failed_files)

    The consumers leave out every detection whose file is in failed_files, so the
    predictions leave out all of those detections too, including ones that loaded
    (e.g. other frames of a video with one unreadable frame). Otherwise the counts
    wouldn't match.
    '''
    filepaths = filepaths.reset_index(drop=True)
    if by_row.shape[1] == 0:  # nothing could be loaded
        loaded = np.zeros(len(by_row), dtype=bool)
    else:
        loaded = ~np.isnan(by_row).any(axis=1)
    failed_files = filepaths[~loaded].unique().tolist()
    keep = ~filepaths.isin(failed_files).to_numpy()
    if failed_files:
        print(f"Warning: {int((~loaded).sum())} detections from {len(failed_files)} files could not be loaded.",
              f"All {int((~keep).sum())} detections from those files are left out of the predictions.")
    return by_row[keep], failed_files


def _failed_files_path(out_file) -> Path:
    return Path(out_file).parent / FAILED_FILES_NAME


def _save_failed_files(out_file, failed_files: list) -> None:
    path = _failed_files_path(out_file)
    if failed_files:
        path.write_text("".join(f"{f}\n" for f in failed_files))
    elif path.is_file():
        path.unlink()  # from an earlier run whose predictions were just overwritten


def _load_failed_files(out_file) -> list:
    path = _failed_files_path(out_file)
    return path.read_text().splitlines() if path.is_file() else []


def _unpack_predictions(predictions_output, failed_files):
    '''(predictions, failed_files) from classify's tuple, or a bare predictions array'''
    if isinstance(predictions_output, (list, tuple)) and len(predictions_output) == 2:
        return predictions_output
    return predictions_output, failed_files


def _remove_failed(animals: pd.DataFrame, predictions_raw, failed_files, file_col: str) -> pd.DataFrame:
    '''Leave out detections from files that failed to load, and check the predictions match'''
    if failed_files is not None and len(failed_files) > 0:
        print(f"Warning: {len(failed_files)} files failed to load during classification",
              " and will be excluded from results.")
        animals = animals[~animals[file_col].isin(failed_files)]
    animals = animals.reset_index(drop=True)
    if len(predictions_raw) != len(animals):
        raise ValueError(f"Number of predictions ({len(predictions_raw)}) does not match number of "
                         f"animal detections ({len(animals)}) after removing failed files.")
    return animals


def single_classification(animals: pd.DataFrame,
                          empty: Optional[pd.DataFrame],
                          predictions_output: Union[np.array, tuple],
                          class_list: Union[list, pd.Series],
                          best: bool = False,
                          count: bool = False,
                          file_col: str = "filepath",
                          failed_files: Optional[list] = None):
    """
    Get maximum likelihood prediction from softmaxed logits.

    Args:
        animals (pd.DataFrame): animal detections from manifest
        empty (Optional[pd.DataFrame]): empty detections from manifest
        predictions_output (Union[np.array, tuple]): softmaxed logits from classify()
            and optionally list of failed files from classify
        class_list (Union[list, pd.Series]): class list associated with model
        best (bool): whether to return one prediction per file
        count (bool): whether to add a count column with number of detections of each species per file
        file_col (str): column name for file paths in the dataframe
        failed_files (Optional[list]): list of files that failed to load during classification

    Returns:
        animals dataframe with "prediction" label an "confidence" columns
    """
    # convert None to empty dataframe fo concat
    if empty is None:
        empty = pd.DataFrame()
    else:
        empty = empty.reset_index(drop=True)
        empty['prediction'] = empty['category_label']
        empty['confidence'] = empty['conf']
        empty['confidence'] = empty['confidence'].replace(np.nan, 1)

    if isinstance(class_list, pd.Series):
        class_list = class_list.to_list()

    # handle tuple output from classify (predictions, failed_files)
    predictions_raw, failed_files = _unpack_predictions(predictions_output, failed_files)

    if not animals.empty:
        animals = _remove_failed(animals, predictions_raw, failed_files, file_col)
    if not animals.empty:
        animals["prediction"] = [class_list[i] for i in np.argmax(predictions_raw, axis=1)]
        animals["confidence"] = animals["conf"].mul(np.max(predictions_raw, axis=1))

    manifest = pd.concat([animals if not animals.empty else None, empty if not empty.empty else None]).reset_index(drop=True)

    # add extension column if not present for video handling
    if 'extension' not in manifest.columns:
        manifest['extension'] = manifest[file_col].apply(lambda x: Path(x).suffix.lower())

    # remove empties from videos
    files = manifest.groupby(file_col)

    for f, file in files:
        if file['extension'].iloc[0] in file_management.VIDEO_EXTENSIONS:
            predictions = file['prediction'].unique()
            if 'empty' in predictions and len(predictions) > 1:
                file = file[file['prediction'] != 'empty']
                # replace empty predictions with most confident non-empty prediction
                top = file.sort_values("confidence", ascending=False).iloc[0]
                cols = ['prediction', 'confidence', 'frame', 'conf', 'max_detection_conf',
                        'category', 'category_label', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
                mask = manifest[file_col] == f
                manifest.loc[mask, cols] = top[cols].values

        # get counts of each prediction for the file if count = True
        if count:
            file['count'] = file['prediction'].map(file['prediction'].value_counts())
            manifest.loc[manifest[file_col] == f, 'count'] = file['count'].values

    # best guess
    if best:
        # take most confident guess
        manifest = manifest.sort_values("confidence", ascending=False)
        manifest = manifest.drop_duplicates(subset=file_col, keep="first")

    return manifest.reset_index(drop=True)


def sequence_classification(animals: pd.DataFrame,
                            empty: Optional[pd.DataFrame],
                            predictions_output: Union[np.array, tuple],
                            class_list: pd.DataFrame,
                            station_col: str,
                            empty_class: str = "",
                            sort_columns: list[str] = None,
                            file_col: str = "filepath",
                            maxdiff: int = 60,
                            failed_files: Optional[list] = None):
    """
    Applies class labels to images based on sequential information.

    This function applies image classifications at a sequence level by leveraging information from
    multiple images. A sequence is defined as all images at the same camera and station where the
    time between consecutive images is <=maxdiff. This can improve classification accuracy, but
    assumes that only one species is present in each sequence. If you regularly expect multiple
    species to occur in an image or sequence don't use this function.

    Args:
        animals (pd.DataFrame): Sub-selection of all images that contain animals
        empty (Optional) (pd.DataFrame): Sub-selection of all images that do not contain animals
        predictions_output (Union[np.array, tuple]): softmaxed logits from classify()
            and optionally list of failed files from classify
        class_list (pd.DataFrame): class list associated with classifier model
        station_col (str): The name of the station column
        empty_class (str) (Optional): the name of class_list 'empty' label
        sort_columns (List of Strings): Defines sorting order for the DataFrame
        file_col (str): The name of the filepath column
        maxdiff (int): Maximum time difference in seconds between any two images in a sequence
        failed_files (Optional[list]): list of files that failed to load during classification

    Returns:
        final_df (pd.DataFrame): Sequence classified data from both animals and empty

    Raises:
        Exception: If 'sort_columns' is not a list or is empty
        Exception: If 'station_col' is not a string or is empty
        Exception: If 'empty' is defined and is not a pandas DataFrame
        Exception: If maxdiff is defined and is not a positive number
    """
    if not isinstance(station_col, str) or station_col == '':
        raise Exception("'station_col' must be a non-empty string")

    # Sanity check to verify that empty is a Pandas DataFrame, if defined
    if empty is not None and not isinstance(animals, pd.DataFrame):
        raise Exception("'empty' must be a DataFrame")

    # Sanity check to verify that maxdiff is a positive number
    if not isinstance(maxdiff, (int, float)) or maxdiff < 0:
        raise Exception("'maxdiff' must be a number >= 0")

    if not {file_col}.issubset(animals.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")

    if not {"datetime"}.issubset(animals.columns):
        raise ValueError("DataFrame must contain 'datetime' column.")

    if "conf" not in animals.columns:
        animals["conf"] = 1

    if empty_class > "":
        empty_col = class_list[class_list == empty_class].index[0]
    else:
        empty_col = None

    # handle tuple output from classify (predictions, failed_files)
    predictions_raw, failed_files = _unpack_predictions(predictions_output, failed_files)
    animals = _remove_failed(animals, predictions_raw, failed_files, file_col)

    if empty is not None or not empty.empty:
        empty["ID"] = range(0, empty.shape[0])
        predempty = empty.pivot(index="ID", columns="prediction", values="confidence")
        # Replace NaN with 0
        predempty = predempty.fillna(0)
        predempty = pd.concat([pd.DataFrame(np.zeros((empty.shape[0], len(class_list)))), predempty], axis=1)

        if empty_class > "":
            if 'empty' in predempty.columns:
                predempty[empty_col] = predempty["empty"]
                predempty = predempty.drop("empty", axis=1)
            class_list = pd.concat([class_list,
                                    pd.Series([x for x in empty["prediction"].unique() if x != "empty"])], ignore_index=True)

        else:
            class_list = pd.concat([class_list, pd.Series(empty["prediction"].unique())], ignore_index=True)
            empty_col = predempty.columns.get_loc("empty")

        # placeholders
        animals["prediction"] = list(class_list[np.argmax(predictions_raw, axis=1)])
        animals["confidence"] = animals["conf"].mul(np.max(predictions_raw, axis=1))

        empty["conf"] = 1
        animals_merged = pd.concat([animals, empty.iloc[:, :-1]]).reset_index(drop=True)  # dont add ID column
        predictions = np.hstack((predictions_raw,
                                 np.zeros((predictions_raw.shape[0], len(predempty.columns) - predictions_raw.shape[1]))))
        # concat
        predictions = np.vstack((predictions, np.array(predempty)))

    if sort_columns is None:
        sort_columns = [station_col, "datetime"]

    animals_merged['datetime'] = pd.to_datetime(animals_merged['datetime'], format="%Y-%m-%d %H:%M:%S")

    sort = animals_merged.sort_values(by=sort_columns).index
    animals_sort = animals_merged.loc[sort].reset_index(drop=True)
    predsort = predictions[sort]

    conf_placeholder = np.zeros(len(animals_sort))
    predict_placeholder = np.empty(len(animals_sort), dtype='U30')
    sequence_placeholder = np.zeros(len(animals_sort))

    i = 0
    s = 0
    while i < len(animals_sort):
        rows = [i]
        last_index = i+1

        while (last_index < len(animals_sort) and not pd.isna(animals_sort.loc[i, "datetime"]) and
               not pd.isna(animals_sort.loc[last_index, "datetime"]) and
               animals_sort.loc[last_index, station_col] == animals_sort.loc[i, station_col] and
               (animals_sort.loc[last_index, "datetime"] - animals_sort.loc[i, "datetime"]).total_seconds() <= maxdiff):
            rows.append(last_index)
            last_index += 1

        rows = np.array(rows)

        # multiple detections in sequence
        if len(rows) > 1:
            predclass = np.argmax(predsort[rows], axis=1)

            # no empties
            if empty_col is None or empty_col not in predclass:
                predsort_confidence = predsort[rows] * np.reshape(animals_sort.loc[rows, 'conf'].values, (-1, 1))
                predbest = np.mean(predsort_confidence, axis=0)
                conf_placeholder[rows] = np.max(predsort_confidence[:, np.argmax(predbest)])
                predict_placeholder[rows] = class_list[np.argmax(predbest)]
                sequence_placeholder[rows] = int(s)

            else:
                mask = pd.DataFrame((predclass == empty_col))
                filtered_animals = animals_sort.loc[rows, file_col].reset_index(drop=True)
                sum_values = mask.groupby(filtered_animals).sum()
                count_values = mask.groupby(filtered_animals).count()

                sel_all_empty = count_values == sum_values
                sel_mixed = np.where(filtered_animals.isin(sel_all_empty[~sel_all_empty[0]].index))[0]
                sel_no_empties = np.where(filtered_animals.isin(sel_all_empty[~sel_all_empty[0]].index) & (predclass != empty_col))[0]

                if len(sel_mixed) > 0 and len(sel_no_empties) > 0:
                    predsort_confidence = predsort[rows[sel_no_empties]] * np.reshape(animals_sort.loc[rows[sel_no_empties], 'conf'].values, (-1, 1))
                    predbest = np.mean(predsort_confidence, axis=0)
                    conf_placeholder[rows[sel_mixed]] = np.max(predsort_confidence[:, np.argmax(predbest)])
                    predict_placeholder[rows[sel_mixed]] = class_list[np.argmax(predbest)]
                    sequence_placeholder[rows[sel_mixed]] = int(s)
                for file in sel_all_empty[sel_all_empty[0]].index:
                    empty_row = np.where(animals_sort[file_col] == file)
                    predsort_confidence = predsort[empty_row] * np.reshape(animals_sort.loc[empty_row, 'conf'].values, (-1, 1))
                    predbest = np.mean(predsort_confidence, axis=0)
                    conf_placeholder[empty_row] = np.max(predsort_confidence[:, np.argmax(predbest)])
                    sequence_placeholder[empty_row] = int(s)
                    predict_placeholder[empty_row] = class_list[np.argmax(predbest)]
        # single row in sequence
        else:
            predbest = predsort[rows]
            conf_placeholder[rows] = np.max(predbest * animals_sort.loc[rows, 'conf'].values)
            predict_placeholder[rows] = class_list[np.argmax(predbest)]
            sequence_placeholder[rows] = int(s)

        i = last_index
        s += 1

    animals_sort['confidence'] = conf_placeholder
    animals_sort['prediction'] = predict_placeholder
    animals_sort['sequence'] = sequence_placeholder

    return animals_sort
