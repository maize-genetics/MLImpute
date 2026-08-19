import os
from typing import NamedTuple, Optional

import pandas as pd
import numpy as np
import logging


class SampleGamete(NamedTuple):
    """A single gamete: a sample name plus its haplotype/gamete index.

    Per the PS4G spec, a gamete header field is either `<sampleName>`
    (index implicitly 0) or `<sampleName>:<gameteIdx>`.
    """
    sample_name: str
    gamete_idx: int

    def label(self, distinguish: bool = False) -> str:
        """Return the display label for this gamete.

        distinguish=False (default) returns the bare sample name, matching
        historical behavior and BED/labels.bed naming. distinguish=True
        returns "sample:idx" so multiple gametes of the same sample (e.g.
        B73:0 and B73:1) don't collide.
        """
        if distinguish:
            return f"{self.sample_name}:{self.gamete_idx}"
        return self.sample_name


def parse_sample_gamete(field: str) -> SampleGamete:
    """Parse a gamete header field into a SampleGamete.

    Accepts both PS4G-spec forms: "B73" (index defaults to 0) and
    "B73:0" (explicit index). Only a trailing ":<digits>" is treated as an
    index suffix; a sample name that itself contains ':' but isn't followed
    by a pure-digit suffix is kept whole.
    """
    if ":" in field:
        name, _, suffix = field.rpartition(":")
        if suffix.isdigit():
            return SampleGamete(name, int(suffix))
    return SampleGamete(field, 0)


def parse_gamete_header_line(line: str) -> Optional[tuple]:
    """Parse a line already known — by its position under the "#gamete" tag
    — to be a gamete record, e.g. "#B73\t0\t784970" or "#B73:0\t0\t784970".

    Returns (SampleGamete, gamete_index, read_count), or None if `line` is
    malformed (not a shape sniff: the caller has already established this
    line *should* be a record — see `read_ps4g_header`). Also rejects
    non-gamete header lines it might be called on directly (e.g.
    "#gamete\tgameteIndex\tcount", "#version=2.0", "#Command: ...",
    "#PS4G") as defense in depth.
    """
    if not line.startswith("#") or "\t" not in line:
        return None
    fields = line[1:].split("\t")
    if len(fields) < 3:
        return None
    gamete_full, idx_str, count_str = fields[0], fields[1], fields[2]
    try:
        idx = int(idx_str)
        count = int(count_str)
    except ValueError:
        return None
    return parse_sample_gamete(gamete_full), idx, count


def _is_gamete_section_tag(line: str) -> bool:
    """True if `line` is the "#gamete\tgameteIndex\tcount" tag that opens
    the gamete section (see `read_ps4g_header`). Only the first tab field is
    checked — the column names after it are informational — so this matches
    regardless of exact header spelling. Case-insensitive and tolerant of
    extra leading '#'s (mirroring the "#PS4G"/"##PS4G" magic line).
    """
    return line.lstrip("#").split("\t", 1)[0].strip().lower() == "gamete"


def _parse_metadata_key(line: str, metadata: dict) -> bool:
    """Consume a keyed header line (##filename1=, #version=, #Command:,
    #TotalUniqueCounts:) into `metadata`. Returns True if `line` was one of
    these, so the caller knows not to also try treating it as a gamete
    record — these keys take precedence even inside the gamete section (a
    producer-declared #TotalUniqueCounts: line interleaved among gamete
    records shouldn't be mistaken for a malformed one).
    """
    if line.startswith("##filename1="):
        metadata["filename1"] = line.split("=", 1)[1]
    elif line.startswith("#version="):
        metadata["version"] = line.split("=", 1)[1]
    elif line.startswith("#Command:"):
        metadata["command"] = line.replace("#Command: ", "")
    elif line.startswith("#TotalUniqueCounts:"):
        metadata["total_reads"] = int(line.split(":", 1)[1])
    else:
        return False
    return True


def _synthesize_gametes_from_data(ps4g_file):
    """Fallback for a file with no "#gamete" section: derive one gamete per
    distinct index seen in the data section's gameteSet column, named by
    that index — rather than falling back to shape-sniffing '#' header
    lines, which is the thing this section-gated design replaces.
    """
    tally = {}
    with open(ps4g_file, "r") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("gameteSet"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            gamete_set_str, count_str = fields[0], fields[3]
            try:
                count = int(count_str)
            except ValueError:
                continue
            for idx_str in gamete_set_str.split(","):
                idx_str = idx_str.strip()
                if not idx_str:
                    continue
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                tally[idx] = tally.get(idx, 0) + count

    return [
        {
            "gamete": str(idx),
            "gamete_index": idx,
            "read_count": tally[idx],
            "sample_gamete": SampleGamete(str(idx), 0),
        }
        for idx in sorted(tally)
    ]


def read_ps4g_header(ps4g_file, distinguish_gametes: bool = False):
    """Read a PS4G file's leading header block in a single pass.

    Gamete records are recognized by position — only "#"-prefixed lines
    between the "#gamete" tag and the end of the header block — never by
    their column shape, so a stray "#"-prefixed comment with the same
    3-column shape (before the tag, or anywhere after the header ends) is
    never mistaken for one. If no "#gamete" tag is found, gametes are
    synthesized from the indices actually used in the data section's
    gameteSet column (see `_synthesize_gametes_from_data`), named by index.

    Returns (metadata, gamete_data) — the same shapes `extract_metadata` and
    `parse_gamete_records` returned previously.
    """
    metadata = {
        "sample_name": os.path.basename(ps4g_file),
        "version": None,
        "filename1": None,
        "command": None,
        "total_reads": None,
    }
    gamete_data = []
    in_gamete_section = False

    with open(ps4g_file, "r") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break  # first non-'#' line ends the header block

            if _is_gamete_section_tag(line):
                in_gamete_section = True
                continue
            if _parse_metadata_key(line, metadata):
                continue
            if not in_gamete_section:
                continue

            parsed = parse_gamete_header_line(line)
            if parsed is None:
                # Malformed record inside the section: skipped, not fatal,
                # section stays open so later well-formed records still parse.
                logging.warning(
                    "Skipping malformed gamete record in %s: %r", ps4g_file, line
                )
                continue
            sample_gamete, idx, count = parsed
            gamete_data.append({
                "gamete": sample_gamete.label(distinguish_gametes),
                "gamete_index": idx,
                "read_count": count,
                "sample_gamete": sample_gamete,
            })

    if not gamete_data:
        gamete_data = _synthesize_gametes_from_data(ps4g_file)

    total_reads = metadata["total_reads"]
    for entry in gamete_data:
        entry["weight"] = entry["read_count"] / total_reads if total_reads else 0.0

    return metadata, gamete_data


def parse_gamete_records(ps4g_file, distinguish_gametes: bool = False):
    """Return one dict per gamete in a PS4G file's gamete section:
    {"gamete", "gamete_index", "read_count", "weight", "sample_gamete"}.

    `distinguish_gametes` controls whether "gamete" is the bare sample name
    (False, default) or "sample:idx" (True) — see SampleGamete.label. See
    `read_ps4g_header` for how gamete records are recognized.
    """
    return read_ps4g_header(ps4g_file, distinguish_gametes)[1]


def convert_ps4g(ps4g_file, weight_strat, collapse):
    """
    This function converts PS4G file into a multihot encoded matrix with optional weighting and collapsing.

    Args:
        ps4g_file (str): Path to the PS4G file.
        weight_strat (str): Weighting strategy, can be 'global', 'read', or 'unweighted'.
        collapse (bool): If True, collapses the gamete sets into a single row per position.

    Returns:
        numpy.ndarray: A multihot encoded matrix representing the PS4G data.
    """

    ps4g_data = load_ps4g_file(ps4g_file)
    metadata, gamete_data = extract_metadata(ps4g_file)

    input_matrix, weights = create_multihot_matrix(ps4g_data, gamete_data, weight_strat, collapse)
    logging.info(f"Converted PS4G file {ps4g_file} with weight '{weight_strat}' and collapse={collapse}.")
    return input_matrix,weights

def decode_position(encoded_pos):
    """
    Decode a 32-bit integer that packs:
      • the upper-8 bits → an index (0-255)
      • the lower-24 bits → a position, but quantised in bins of 256 bp
    This is lossy because we multiplied by 256 during encoding.
    """
    idx = (encoded_pos >> 24) & 0xFF           # top-byte index (unsigned)
    pos = (encoded_pos & 0x0FFFFFF) * 256      # restore to bp units
    return idx, pos

def load_ps4g_file(ps4g_file):
    """
    Load a PS4G file and return the DataFrame.

    Args:
        ps4g_file (str): Path to the PS4G file.

    Returns:
        pd.DataFrame: DataFrame containing the PS4G data.
    """
    logging.info(f"Loading PS4G file {ps4g_file}")
    ps4g = pd.read_csv(ps4g_file, delimiter="\t", comment="#")
    ps4g['gameteSet'] = ps4g['gameteSet'].apply(lambda x: list(map(int, x.split(','))))
    return ps4g

def extract_metadata(ps4g_file, distinguish_gametes: bool = False):
    """
    Extract metadata from the PS4G file.

    Args:
        ps4g_file (str): Path to the PS4G file.
        distinguish_gametes (bool): If True, gamete names include the ":idx"
            suffix (e.g. "B73:0" vs "B73:1"); if False (default), gametes
            are named by sample only, matching historical behavior.

    Returns:
        dict: A dictionary containing metadata such as sample name, version, filename, command, and total reads.
        list: A list of dictionaries containing gamete data with gamete name, index, read count, and weight.
    """
    logging.info(f"Extracting metadata from PS4G file {ps4g_file}")
    return read_ps4g_header(ps4g_file, distinguish_gametes)

def create_multihot_matrix(ps4g, gamete_data, weight_strat, collapse):
    """
    Create a multihot encoded matrix from the PS4G data.

    Args:
        ps4g (pd.DataFrame): DataFrame containing the PS4G data.
        gamete_data (list): List of dictionaries containing gamete data.
        weight_strat (str): Weighting strategy.
        collapse (bool): If True, collapses the gamete sets.

    Returns:
        np.ndarray: A multihot encoded matrix.
    """
    logging.info("Creating multihot matrix from PS4G data")
    # Get number of unique gametes
    gamete_indices = [entry["gamete_index"] for entry in gamete_data]
    num_classes = len(gamete_indices)

    # Create unique position identifiers using refContig and refPosBinned
    ps4g['position_id'] = ps4g['refContig'].astype(str) + '_' + ps4g['refPosBinned'].astype(str)
    unique_positions = ps4g['position_id'].unique()
    pos_to_idx = {pos: i for i, pos in enumerate(unique_positions)}

    if collapse:
        logging.info("Collapsing")
        X_multihot, collapsed_df = collapse_ps4g(num_classes, ps4g, unique_positions)

    else:
        logging.info("not collapsed")
        X_multihot = np.zeros((len(ps4g), num_classes), dtype=np.float32)

        for i, indices in enumerate(ps4g['gameteSet']):
            X_multihot[i, indices] = 1  # vectorized assignment


    if weight_strat == "global":
        input_matrix, weights = process_global_weight_mode(X_multihot, gamete_data, num_classes)
    else:
        logging.info("unweighted")
        input_matrix = X_multihot
        weights = np.ones(num_classes, dtype=np.float32)


    return input_matrix, weights

def process_global_weight_mode(X_multihot, gamete_data, num_classes):
    """
    Process the global weight mode for the multihot encoded matrix.
    Args:
        X_multihot (np.ndarray): The multihot encoded matrix.
        gamete_data (list): List of dictionaries containing gamete data.
        num_classes (int): Number of unique gametes.
    Returns:
        np.ndarray: The input matrix with global weights applied.
    """
    logging.info("global weight")
    # Map index → weight
    index_to_weight = {entry["gamete_index"]: entry["weight"] for entry in gamete_data}
    # Create aligned weight vector for all columns
    global_weights = np.array([index_to_weight.get(i, 0.0) for i in range(num_classes)], dtype=np.float32)

    return X_multihot, global_weights

def collapse_ps4g(num_classes, ps4g, unique_positions):
    """ Collapse the PS4G DataFrame by position and aggregate gamete sets.
    Args:
        num_classes (int): Number of unique gametes.
        ps4g (pd.DataFrame): DataFrame containing the PS4G data with position_id column.
        unique_positions (np.ndarray): Array of unique position identifiers.
    Returns:
        np.ndarray: A multihot encoded matrix with collapsed gamete sets.
        pd.DataFrame: DataFrame containing collapsed PS4G data with aggregated counts.
    """
    collapsed_df = ps4g.groupby('position_id').agg({
        'refContig': 'first',
        'refPosBinned': 'first',
        'gameteSet': lambda x: sorted(set().union(*x)),
        'count': 'sum'
    }).reset_index()
    collapsed_df = collapsed_df.set_index('position_id').loc[unique_positions].reset_index()
    X_multihot = np.zeros((len(collapsed_df), num_classes), dtype=np.float32)
    for i, indices in enumerate(collapsed_df['gameteSet']):
        X_multihot[i, indices] = 1  # vectorized assignment
    return X_multihot, collapsed_df

def build_index_lookup(ps4g_file, distinguish_gametes: bool = False):
    """
    Build an array mapping gamete_index -> gamete name from a PS4G file's
    metadata lines.

    Args:
        ps4g_file (str): Path to the PS4G file.
        distinguish_gametes (bool): If True, names include the ":idx" suffix
            (e.g. "B73:0" vs "B73:1"); if False (default), gametes are named
            by sample only.
    """
    gamete_data = parse_gamete_records(ps4g_file, distinguish_gametes)
    if not gamete_data:
        raise ValueError(f"No gamete metadata lines found in PS4G file: {ps4g_file}")
    index_to_name = {entry["gamete_index"]: entry["gamete"] for entry in gamete_data}
    max_index = max(index_to_name.keys())  # ensure all indices fit
    index_array = [index_to_name[i] for i in range(max_index + 1)]
    return index_array
