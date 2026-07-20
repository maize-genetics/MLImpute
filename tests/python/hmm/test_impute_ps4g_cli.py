import pytest

from python.hmm.impute_ps4g import main


def write_ps4g_file(path, gamete_lines, data_lines):
    total_counts = sum(int(d[3]) for d in data_lines)
    with open(path, "w") as fh:
        fh.write("#PS4G\n")
        fh.write("#version=2.0\n")
        fh.write("#Command: test\n")
        fh.write(f"#TotalUniqueCounts: {total_counts}\n")
        fh.write("#gamete\tgameteIndex\tcount\n")
        for name, idx, count in gamete_lines:
            fh.write(f"#{name}\t{idx}\t{count}\n")
        fh.write("gameteSet\trefContig\trefPosBinned\tcount\n")
        for gamete_set, contig, pos, count in data_lines:
            fh.write(f"{gamete_set}\t{contig}\t{pos}\t{count}\n")
    return str(path)


def write_keyfile(path, sample_name, ps4g_path):
    with open(path, "w") as fh:
        fh.write("sampleName\tfilename\n")
        fh.write(f"{sample_name}\t{ps4g_path}\n")
    return str(path)


TWO_GAMETES = [("lineA:0", 0, 0), ("lineB:0", 1, 0)]


class TestHaploidCLI:
    def test_single_gamete_calls_it_everywhere(self, tmp_path):
        # All reads support gamete 0 (lineA:0) at every position - the HMM should
        # assign the whole chromosome to lineA:0 (mirrors
        # ImputePathFromPs4gTest.testImputeHaploidPathOutput).
        data_lines = [("0", "chr1", bin_pos, 10) for bin_pos in range(1, 6)]
        ps4g_file = write_ps4g_file(tmp_path / "singleGamete.ps4g", TWO_GAMETES, data_lines)
        keyfile = write_keyfile(tmp_path / "key.txt", "sampleX", ps4g_file)
        out_dir = tmp_path / "haploidOut"

        main([
            "--path-keyfile", str(keyfile),
            "--out-path-dir", str(out_dir),
            "--prob-correct", "0.99",
            "--prob-same", "0.9999",
            "--bin-size", "1",
        ])

        bed_file = out_dir / "sampleX_imputed_path.bed"
        assert bed_file.exists()
        lines = bed_file.read_text().splitlines()
        assert lines[0] == "chrom\tstart\tend\tparent1"
        data_rows = lines[1:]
        assert data_rows
        for line in data_rows:
            cols = line.split("\t")
            assert cols[0] == "chr1"
            assert cols[3] == "lineA:0"

    def test_bed_midpoint_coordinates(self, tmp_path):
        # bins 10,20,30,40,50 with bin-size 100 - verify the midpoint arithmetic
        # exactly (mirrors ImputePathFromPs4gTest.testImputeHaploidPathBedCoordinates).
        bins = [10, 20, 30, 40, 50]
        data_lines = [("0", "chr1", b, 5) for b in bins]
        ps4g_file = write_ps4g_file(tmp_path / "coordTest.ps4g", TWO_GAMETES, data_lines)
        keyfile = write_keyfile(tmp_path / "coordKey.txt", "coordSample", ps4g_file)
        out_dir = tmp_path / "coordOut"

        main([
            "--path-keyfile", str(keyfile),
            "--out-path-dir", str(out_dir),
            "--prob-correct", "0.99",
            "--prob-same", "0.9999",
            "--bin-size", "100",
        ])

        lines = (out_dir / "coordSample_imputed_path.bed").read_text().splitlines()[1:]
        assert len(lines) == 5

        expected = [
            (1, 1500),
            (1501, 2500),
            (2501, 3500),
            (3501, 4500),
            (4501, 5000),
        ]
        for row, (start, end) in zip(lines, expected):
            cols = row.split("\t")
            assert int(cols[1]) == start
            assert int(cols[2]) == end

    def test_recombination_switches_path(self, tmp_path):
        # First 4 bins support lineA:0, last 4 support lineB:0; with low prob-same
        # the Viterbi path should switch partway through (mirrors
        # ImputePathFromPs4gTest.testImputeHaploidPathWithRecombination).
        data_lines = [("0", "chr1", b, 20) for b in range(1, 5)] + [("1", "chr1", b, 20) for b in range(5, 9)]
        ps4g_file = write_ps4g_file(tmp_path / "recombTest.ps4g", TWO_GAMETES, data_lines)
        keyfile = write_keyfile(tmp_path / "recombKey.txt", "recombSample", ps4g_file)
        out_dir = tmp_path / "recombOut"

        main([
            "--path-keyfile", str(keyfile),
            "--out-path-dir", str(out_dir),
            "--prob-correct", "0.99",
            "--prob-same", "0.5",
            "--bin-size", "1",
        ])

        lines = (out_dir / "recombSample_imputed_path.bed").read_text().splitlines()[1:]
        assert len(lines) == 8
        parents = [line.split("\t")[3] for line in lines]
        assert parents[:4] == ["lineA:0"] * 4
        assert parents[4:] == ["lineB:0"] * 4

    def test_multi_contig_runs_independently(self, tmp_path):
        # mirrors ImputePathFromPs4gTest.testImputeHaploidPathMultiContig
        data_lines = [
            ("0", "chr1", 1, 10),
            ("0", "chr1", 2, 10),
            ("1", "chr2", 1, 10),
            ("1", "chr2", 2, 10),
        ]
        ps4g_file = write_ps4g_file(tmp_path / "multiContig.ps4g", TWO_GAMETES, data_lines)
        keyfile = write_keyfile(tmp_path / "multiContigKey.txt", "multiSample", ps4g_file)
        out_dir = tmp_path / "multiContigOut"

        main([
            "--path-keyfile", str(keyfile),
            "--out-path-dir", str(out_dir),
            "--prob-correct", "0.99",
            "--prob-same", "0.9999",
            "--bin-size", "1",
        ])

        lines = (out_dir / "multiSample_imputed_path.bed").read_text().splitlines()[1:]
        assert len(lines) == 4
        chr1_rows = [line for line in lines if line.startswith("chr1\t")]
        chr2_rows = [line for line in lines if line.startswith("chr2\t")]
        assert len(chr1_rows) == 2
        assert len(chr2_rows) == 2
        assert all(row.split("\t")[3] == "lineA:0" for row in chr1_rows)
        assert all(row.split("\t")[3] == "lineB:0" for row in chr2_rows)

    def test_read_file_single_input(self, tmp_path):
        data_lines = [("0", "chr1", b, 10) for b in range(1, 4)]
        ps4g_file = write_ps4g_file(tmp_path / "readFileSample.ps4g", TWO_GAMETES, data_lines)
        out_dir = tmp_path / "readFileOut"

        main(["--read-file", str(ps4g_file), "--out-path-dir", str(out_dir), "--bin-size", "1"])

        bed_file = out_dir / "readFileSample_imputed_path.bed"
        assert bed_file.exists()


class TestDiploidCLI:
    def test_inbred_diploid_is_homozygous(self, tmp_path):
        # With inbreedCoef=1.0 and all reads supporting gamete 0, the diploid path
        # is forced homozygous (lineA:0, lineA:0) throughout (mirrors
        # ImputePathFromPs4gTest.testImputeDiploidPathOutput).
        data_lines = [("0", "chr1", b, 10) for b in range(1, 6)]
        ps4g_file = write_ps4g_file(tmp_path / "diploidTest.ps4g", TWO_GAMETES, data_lines)
        keyfile = write_keyfile(tmp_path / "diploidKey.txt", "diploidSample", ps4g_file)
        out_dir = tmp_path / "diploidOut"

        main([
            "--path-keyfile", str(keyfile),
            "--out-path-dir", str(out_dir),
            "--path-type", "diploid",
            "--prob-correct", "0.99",
            "--prob-same", "0.9999",
            "--inbreed-coef", "1.0",
            "--bin-size", "1",
        ])

        txt_file = out_dir / "diploidSample_imputed_path.txt"
        assert txt_file.exists()
        lines = txt_file.read_text().splitlines()
        assert lines[0] == "chrom\tstart\tend\tparent1\tparent2"
        data_rows = lines[1:]
        assert data_rows
        for row in data_rows:
            cols = row.split("\t")
            assert len(cols) == 5
            assert cols[0] == "chr1"
            assert cols[3] == "lineA:0"
            assert cols[4] == "lineA:0"

    def test_diploid_with_n_parents_restriction(self, tmp_path):
        # lineA:0 and lineB:0 each have strong *independent* support (never sharing
        # a gameteSet), so most_likely_parents reliably restricts to {0,1} over the
        # rare, independent lineC:0 evidence.
        gametes = [("lineA:0", 0, 0), ("lineB:0", 1, 0), ("lineC:0", 2, 0)]
        data_lines = (
            [("0", "chr1", b, 10) for b in range(1, 4)]
            + [("1", "chr1", b, 10) for b in range(4, 7)]
            + [("2", "chr1", b, 1) for b in range(7, 10)]
        )
        ps4g_file = write_ps4g_file(tmp_path / "nParents.ps4g", gametes, data_lines)
        keyfile = write_keyfile(tmp_path / "nParentsKey.txt", "nParentsSample", ps4g_file)
        out_dir = tmp_path / "nParentsOut"

        main([
            "--path-keyfile", str(keyfile),
            "--out-path-dir", str(out_dir),
            "--path-type", "diploid",
            "--n-parents", "2",
            "--bin-size", "1",
        ])

        txt_file = out_dir / "nParentsSample_imputed_path.txt"
        assert txt_file.exists()
        rows = txt_file.read_text().splitlines()[1:]
        assert rows
        for row in rows:
            cols = row.split("\t")
            assert cols[3] in ("lineA:0", "lineB:0")
            assert cols[4] in ("lineA:0", "lineB:0")


class TestCLIValidation:
    def test_missing_input_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--out-path-dir", str(tmp_path)])

    def test_missing_output_dir_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--read-file", "someFile.ps4g"])

    def test_bad_path_type_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--read-file", "someFile.ps4g", "--out-path-dir", str(tmp_path), "--path-type", "invalid"])
