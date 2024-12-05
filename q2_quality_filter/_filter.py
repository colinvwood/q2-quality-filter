# ----------------------------------------------------------------------------
# Copyright (c) 2017-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum
import gzip
import os
from typing import Union
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from q2_types.per_sample_sequences import (
    SingleLanePerSampleSingleEndFastqDirFmt,
    SingleLanePerSamplePairedEndFastqDirFmt,
    FastqManifestFormat,
    YamlFormat,
)

from q2_quality_filter._format import _ReadDirectionUnion


@dataclass
class FastqRecord:
    sequence_header: bytes
    sequence: bytes
    quality_header: bytes
    quality_scores: bytes


def _read_fastq_records(filepath: str):
    '''
    A generator for a fastq file that yields sequence records. The fastq file
    is assumed to be gzipped.

    Parameters
    ----------
    filepath : str
        The filepath to the fastq.gz file.

    Yields
    ------
    SequenceRecord
        A sequence record representing a record from the fastq file.
    '''
    fh = gzip.open(filepath, 'rb')
    while True:
        try:
            sequence_header = next(fh)
            sequence = next(fh)
            quality_header = next(fh)
            quality_scores = next(fh)
        except StopIteration:
            fh.close()
            break

        yield FastqRecord(
            sequence_header.strip(),
            sequence.strip(),
            quality_header.strip(),
            quality_scores.strip()
        )


def _find_low_quality_window(
    quality_scores: bytes,
    phred_offset: int,
    min_quality: int,
    window_length: int
) -> int | None:
    '''
    Searches a sequence of quality scores for subsequences (windows) of length
    `window_length` that consist of quality scores each less than
    `min_quality`. If one or more such windows exist then the index of the
    first position of the first such window is returned (which will be the
    truncation position). Otherwise None is returned.

    Parameters
    ----------
    quality_scores : bytes
        The quality scores byte string for a fastq record.
    phred_offset : int
        The PHRED offset encoding of the quality scores.
    min_quality : int
        The minimum quality that a base must have in order to not be considered
        part of a low quality window.
    window_length : int
        The length of the low quality window to search for.

    Returns
    -------
    int or None
        The index of the first position of the first low quality window found
        or None if no such window is found.

    '''
    # parse and adjust quality scores
    quality_scores_parsed = np.frombuffer(
        quality_scores, np.uint8
    )
    quality_scores_adjusted = quality_scores_parsed - phred_offset
    less_than_min_quality = quality_scores_adjusted < min_quality

    # use a convolution to detect bad quality windows
    window = np.ones(window_length, dtype=int)
    convolution = np.convolve(less_than_min_quality, window, mode='valid')
    window_indices = np.where(convolution == window_length)[0]

    if len(window_indices) == 0:
        return None

    return window_indices[0]


def _truncate(fastq_record: FastqRecord, position: int) -> FastqRecord:
    '''
    Truncates a fastq record's sequence and quality scores to a specified
    `position`. Note that `position` is the first position that is excluded
    from the resulting record.

    Parameters
    ----------
    fastq_record : FastqRecord
        The fastq record to truncate
    position : int
        The truncation position

    Returns
    -------
    FastqRecord
        The truncated fastq record.
    '''
    fastq_record.sequence = fastq_record.sequence[:position]
    fastq_record.quality_scores = fastq_record.quality_scores[:position]

    return fastq_record


class RecordStatus(Enum):
    UNTRUNCATED = 1
    TRUNCATED = 2
    SHORT = 3
    AMBIGUOUS = 4
    TRUNCATED_AMBIGUOUS = 5


def _process_record(
    fastq_record: FastqRecord,
    phred_offset: int,
    min_quality: int,
    window_length: int,
    min_length_fraction: float,
    max_ambiguous: int,
) -> tuple[FastqRecord, RecordStatus]:
    '''
    Processes a fastq record by detecting low quality windows, truncating if
    one or more such windows are found, detecting if a truncated record is too
    short, and finally detecting if the number of ambiguous bases is too high.

    Parameters
    ----------
    fastq_record : FastqRecord | None
        The fastq record to be processed. None if record does not exist (for
        convenience when reverse reads are not present).
    phred_offset : int
        The PHRED encoding of the record's quality scores.
    min_quality : int
        The minimum quality that a base must have in order to not be considered
        part of a low quality window.
    window_length : int
        The length of the low quality window to search for.
    min_length_fraction : float
        The fraction of its original length a record must be greater than to
        be retained.
    max_ambiguous : int
        The maximum number of ambiguous bases a record may contain to be
        retained.

    Returns
    -------
    tuple[FastqRecord, RecordStatus]
        A tuple containing the processed record and its status.
    '''
    if fastq_record is None:
        return None, None

    status = RecordStatus.UNTRUNCATED

    # search for low quality window
    truncation_position = _find_low_quality_window(
        fastq_record.quality_scores,
        phred_offset,
        min_quality,
        window_length
    )

    # check if truncation should be performed mark short if necessary
    initial_record_length = len(fastq_record.sequence)
    if truncation_position is not None:
        fastq_record = _truncate(fastq_record, truncation_position)
        status = RecordStatus.TRUNCATED

        trunc_fraction = truncation_position / initial_record_length
        if trunc_fraction <= min_length_fraction:
            status = RecordStatus.SHORT

            return fastq_record, status

    # mark ambiguous if too many ambiguous bases are present
    if fastq_record.sequence.count(b'N') > max_ambiguous:
        if status == RecordStatus.TRUNCATED:
            status = RecordStatus.TRUNCATED_AMBIGUOUS
        else:
            status = RecordStatus.AMBIGUOUS

    return fastq_record, status


def _is_retained(
    forward_status: RecordStatus,
    reverse_status: RecordStatus | None,
    filtering_stats_df: pd.DataFrame,
    sample_id: str
) -> bool:
    '''
    Determines whether a fastq record or pair of fastq records will retained
    in the output. The `reverse_status` is None in the case of single-end
    reads.

    Parameters
    ----------
    forward_status : RecordStatus
        The status of the record from the forward fastq file.
    reverse_status : RecordStatus or None
        The status of the record from the reverse fastq file if it exists
        otherwise None.
    filtering_stats_df : pd.DataFrame
        The data structure that tracks filtering stats.
    sample_id : str
        The sample id that the record(s) belongs to.

    Returns
    -------
    bool
        True if the record(s) is to be retained, False otherwise.
    '''
    filtering_stats_df.loc[sample_id, 'total-input-reads'] += 1

    if (RecordStatus.SHORT in (forward_status, reverse_status)):
        filtering_stats_df.loc[sample_id, 'reads-truncated'] += 1
        filtering_stats_df.loc[
            sample_id, 'reads-too-short-after-truncation'
        ] += 1
        return False

    if (RecordStatus.AMBIGUOUS in (forward_status, reverse_status)):
        filtering_stats_df.loc[
            sample_id, 'reads-exceeding-maximum-ambiguous-bases'
        ] += 1
        return False

    if (RecordStatus.TRUNCATED_AMBIGUOUS in (forward_status, reverse_status)):
        filtering_stats_df.loc[sample_id, 'reads-truncated'] += 1
        filtering_stats_df.loc[
            sample_id, 'reads-exceeding-maximum-ambiguous-bases'
        ] += 1
        return False

    if (RecordStatus.TRUNCATED in (forward_status, reverse_status)):
        filtering_stats_df.loc[sample_id, 'reads-truncated'] += 1

    filtering_stats_df.loc[sample_id, 'total-retained-reads'] += 1

    return True


def _align_records(
    forward_record: FastqRecord, reverse_record: FastqRecord
) -> tuple[FastqRecord, FastqRecord]:
    '''
    Align a forward record and reverse record to the same truncation length.
    Note that if either (forward or reverse) truncation resulted in the record
    falling below the minimum length fraction then this was already handled
    upstream.

    Parameters
    ----------
    forward_record : FastqRecord
        The record from the forward fastq file.
    reverse_record : FastqRecord
        The record from the reverse fastq file.

    Returns
    -------
    tuple[FastqRecord, FastqRecord]
        The length-aligned forward and reverse records.
    '''
    if len(forward_record.sequence) < len(reverse_record.sequence):
        reverse_record = _truncate(
            reverse_record, len(forward_record.sequence)
        )
    elif len(reverse_record.sequence) < len(forward_record.sequence):
        forward_record = _truncate(
            forward_record, len(reverse_record.sequence)
        )

    return forward_record, reverse_record


def _write_record(fastq_record: FastqRecord, fh: gzip.GzipFile) -> None:
    '''
    Writes a fastq record to an open fastq file.

    Parameters
    ----------
    fastq_record : FastqRecord
        The fastq record to be written.
    fh : GzipFile
        The output fastq file handler.

    Returns
    -------
    None
    '''
    fh.write(fastq_record.sequence_header + b'\n')
    fh.write(fastq_record.sequence + b'\n')
    fh.write(fastq_record.quality_header + b'\n')
    fh.write(fastq_record.quality_scores + b'\n')


def q_score(
    demux: Union[
        SingleLanePerSampleSingleEndFastqDirFmt,
        SingleLanePerSamplePairedEndFastqDirFmt
    ],
    min_quality: int = 4,
    quality_window: int = 3,
    min_length_fraction: float = 0.75,
    max_ambiguous: int = 0
) -> (_ReadDirectionUnion, pd.DataFrame):
    '''
    Parameter defaults as used in Bokulich et al, Nature Methods 2013, same as
    QIIME 1.9.1.
    '''
    # we need to use a union type of single-end and paired-end formats
    # which will be transformed by the framework to the appropriate return type
    union_format = _ReadDirectionUnion()
    union_format.format = type(demux)()

    result = union_format.format

    manifest = FastqManifestFormat()
    manifest_fh = manifest.open()
    manifest_fh.write('sample-id,filename,direction\n')

    if isinstance(result, SingleLanePerSamplePairedEndFastqDirFmt):
        paired = True
    else:
        paired = False

    # load the input demux manifest
    metadata_view = demux.metadata.view(YamlFormat).open()
    phred_offset = yaml.load(metadata_view,
                             Loader=yaml.SafeLoader)['phred-offset']
    demux_manifest = demux.manifest.view(demux.manifest.format)
    demux_manifest_df = pd.read_csv(
        demux_manifest.open(), dtype=str, comment='#'
    )
    demux_manifest_df.set_index('filename', inplace=True)

    filtering_stats_df = pd.DataFrame(
        data=0,
        index=demux_manifest_df['sample-id'],
        columns=[
            'total-input-reads', 'total-retained-reads', 'reads-truncated',
            'reads-too-short-after-truncation',
            'reads-exceeding-maximum-ambiguous-bases'
        ]
    )

    for barcode_id, filename in enumerate(demux_manifest_df.index.values):
        if 'R2' in str(filename):
            # we handle a read pair in the iteration for R1
            continue

        # look up sample id
        sample_id = demux_manifest_df.loc[str(filename)]['sample-id']

        # create path for output fastq file
        forward_path = result.sequences.path_maker(
            sample_id=sample_id,
            barcode_id=barcode_id,
            lane_number=1,
            read_number=2
        )
        if paired:
            reverse_path = result.sequences.path_maker(
                sample_id=sample_id,
                barcode_id=barcode_id,
                lane_number=1,
                read_number=2
            )

        forward_fh = gzip.open(forward_path, mode='wb')
        forward_input_fp = Path(demux.path) / filename

        if paired:
            reverse_fh = gzip.open(reverse_path, mode='wb')
            reverse_input_fp = Path(demux.path) / filename.replace('R1', 'R2')

            forward_iterator = _read_fastq_records(str(forward_input_fp))
            reverse_iterator = _read_fastq_records(str(reverse_input_fp))
            iterator = zip(forward_iterator, reverse_iterator)
        else:
            iterator = _read_fastq_records(str(forward_input_fp))

        for fastq_record in iterator:
            # process record(s)
            if paired:
                forward_record, reverse_record = fastq_record
            else:
                forward_record = fastq_record
                reverse_record = None

            forward_record, forward_status = _process_record(
                fastq_record=forward_record,
                phred_offset=phred_offset,
                min_quality=min_quality,
                window_length=quality_window + 1,
                min_length_fraction=min_length_fraction,
                max_ambiguous=max_ambiguous
            )
            reverse_record, reverse_status = _process_record(
                fastq_record=reverse_record,
                phred_offset=phred_offset,
                min_quality=min_quality,
                window_length=quality_window + 1,
                min_length_fraction=min_length_fraction,
                max_ambiguous=max_ambiguous
            )

            # see if record(s) is retained and update filtering stats
            retained = _is_retained(
                forward_status,
                reverse_status,
                filtering_stats_df,
                sample_id
            )

            # if retained, align truncations and write to output files
            if retained:
                if paired:
                    forward_record, reverse_record = _align_records(
                        forward_record, reverse_record
                    )
                    _write_record(forward_record, forward_fh)
                    _write_record(reverse_record, reverse_fh)
                else:
                    _write_record(forward_record, forward_fh)

        # close output file(s) and update manifest if record(s) is retained,
        # otherwise delete the empty file(s)
        forward_fh.close()
        if paired:
            reverse_fh.close()

        if filtering_stats_df.loc[sample_id, 'total-retained-reads'] > 0:
            manifest_fh.write(f'{sample_id},{forward_path.name},forward\n')
            if paired:
                manifest_fh.write(
                    f'{sample_id},{reverse_path.name},reverse\n'
                )
        else:
            os.remove(forward_path)
            if paired:
                os.remove(reverse_path)

    # error if all samples retained no reads
    if filtering_stats_df['total-retained-reads'].sum() == 0:
        msg = (
            'All sequences from all samples were filtered. The parameter '
            'choices may have been too stringent for the data.'
        )
        raise ValueError(msg)

    # write output manifest and metadata files to format
    manifest_fh.close()
    result.manifest.write_data(manifest, FastqManifestFormat)

    metadata = YamlFormat()
    metadata.path.write_text(yaml.dump({'phred-offset': phred_offset}))
    result.metadata.write_data(metadata, YamlFormat)

    return union_format, filtering_stats_df
