# ----------------------------------------------------------------------------
# Copyright (c) 2017-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from dataclasses import dataclass
import gzip
import os
import yaml
import pandas as pd

import numpy as np
from q2_types.per_sample_sequences import (
            SingleLanePerSampleSingleEndFastqDirFmt,
            FastqManifestFormat, YamlFormat, FastqGzFormat)


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
    demux: SingleLanePerSampleSingleEndFastqDirFmt,
    min_quality: int = 4,
    quality_window: int = 3,
    min_length_fraction: float = 0.75,
    max_ambiguous: int = 0
) -> (SingleLanePerSampleSingleEndFastqDirFmt, pd.DataFrame):
    '''
    Parameter defaults as used in Bokulich et al, Nature Methods 2013, same as
    QIIME 1.9.1.

    Parameters
    ----------

    Returns
    -------
    tuple[SingleLanePerSampleSingleEndFastqDirFmt, pd.DataFrame]

    '''

    # TODO: paired/single handling
    # create the output format and its manifest format
    result = SingleLanePerSampleSingleEndFastqDirFmt()

    manifest = FastqManifestFormat()
    manifest_fh = manifest.open()
    manifest_fh.write('sample-id,filename,direction\n')

    # load the input demux manifest
    metadata_view = demux.metadata.view(YamlFormat).open()
    phred_offset = yaml.load(metadata_view,
                             Loader=yaml.SafeLoader)['phred-offset']
    demux_manifest = demux.manifest.view(demux.manifest.format)
    demux_manifest = pd.read_csv(demux_manifest.open(), dtype=str)
    demux_manifest.set_index('filename', inplace=True)

    # HACK: we have to deal with comment lines that may be present in the
    # manifest that used to be written by this method
    demux_manifest = demux_manifest[
        ~demux_manifest['sample-id'].str.startswith('#')
    ]

    filtering_stats_df = pd.DataFrame(
        data=0,
        index=demux_manifest['sample-id'],
        columns=[
            'total-input-reads', 'total-retained-reads', 'reads-truncated',
            'reads-too-short-after-truncation',
            'reads-exceeding-maximum-ambiguous-bases'
        ]
    )

    iterator = demux.sequences.iter_views(FastqGzFormat)
    for barcode_id, (filename, filepath) in enumerate(iterator):
        sample_id = demux_manifest.loc[str(filename)]['sample-id']

        # barcode ID, lane number and read number are not relevant here
        path = result.sequences.path_maker(
            sample_id=sample_id,
            barcode_id=barcode_id,
            lane_number=1,
            read_number=1
        )

        output_fh = gzip.open(path, mode='wb')

        for fastq_record in _read_fastq_records(str(filepath)):
            filtering_stats_df.loc[sample_id, 'total-input-reads'] += 1

            # search for low quality window
            truncation_position = _find_low_quality_window(
                fastq_record.quality_scores,
                phred_offset,
                min_quality,
                quality_window + 1
            )

            # truncate fastq record if necessary and discard if it has been
            # made too short
            initial_record_length = len(fastq_record.sequence)
            if truncation_position is not None:
                fastq_record = _truncate(fastq_record, truncation_position)
                filtering_stats_df.loc[sample_id, 'reads-truncated'] += 1

                trunc_fraction = truncation_position / initial_record_length
                if trunc_fraction <= min_length_fraction:
                    filtering_stats_df.loc[
                        sample_id, 'reads-too-short-after-truncation'
                    ] += 1
                    continue

            # discard record if there are too many ambiguous bases
            if fastq_record.sequence.count(b'N') > max_ambiguous:
                filtering_stats_df.loc[
                    sample_id, 'reads-exceeding-maximum-ambiguous-bases'
                ] += 1
                continue

            # write record to output file
            _write_record(fastq_record, output_fh)
            filtering_stats_df.loc[sample_id, 'total-retained-reads'] += 1

        # close output file and update manifest if records were retained,
        # otherwise delete the empty file
        output_fh.close()
        if filtering_stats_df.loc[sample_id, 'total-retained-reads'] > 0:
            # TODO
            direction = 'forward'
            manifest_fh.write(f'{sample_id},{path.name},{direction}\n')
        else:
            os.remove(path)

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

    return result, filtering_stats_df
