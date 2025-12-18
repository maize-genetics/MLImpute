import biokotlin.seq.NucSeqRecord
import biokotlin.seqIO.FastaIO
import biokotlin.seqIO.SeqType
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test


class SampleTest {

    /**
     * Test that splitting an empty ignoreContig string produces an empty list.
     * This is a regression test for the bug where "" would become [""],
     * causing all contigs to be ignored (since every string contains "").
     */
    @Test
    fun testIgnoreContigEmptyStringParsing() {
        val emptyIgnoreContig = ""
        val ignoreStrings = emptyIgnoreContig.split(",").filter { it.isNotBlank() }
        
        assertTrue(ignoreStrings.isEmpty(), "Empty ignoreContig should produce empty list")
    }

    /**
     * Test that splitting a blank ignoreContig string (whitespace only) produces an empty list.
     */
    @Test
    fun testIgnoreContigBlankStringParsing() {
        val blankIgnoreContig = "   "
        val ignoreStrings = blankIgnoreContig.split(",").filter { it.isNotBlank() }
        
        assertTrue(ignoreStrings.isEmpty(), "Blank ignoreContig should produce empty list")
    }

    /**
     * Test that valid patterns are correctly parsed.
     */
    @Test
    fun testIgnoreContigValidPatternsParsing() {
        val ignoreContig = "chr1,chr2,scaffold"
        val ignoreStrings = ignoreContig.split(",").filter { it.isNotBlank() }
        
        assertEquals(3, ignoreStrings.size)
        assertEquals(listOf("chr1", "chr2", "scaffold"), ignoreStrings)
    }

    /**
     * Test that patterns with empty entries in between are handled correctly.
     * e.g., "chr1,,chr2" should produce ["chr1", "chr2"], not ["chr1", "", "chr2"]
     */
    @Test
    fun testIgnoreContigWithEmptyEntriesParsing() {
        val ignoreContig = "chr1,,chr2"
        val ignoreStrings = ignoreContig.split(",").filter { it.isNotBlank() }
        
        assertEquals(2, ignoreStrings.size)
        assertEquals(listOf("chr1", "chr2"), ignoreStrings)
    }

    /**
     * Test that patterns with trailing/leading commas are handled correctly.
     */
    @Test
    fun testIgnoreContigWithTrailingLeadingCommasParsing() {
        val ignoreContig = ",chr1,chr2,"
        val ignoreStrings = ignoreContig.split(",").filter { it.isNotBlank() }
        
        assertEquals(2, ignoreStrings.size)
        assertEquals(listOf("chr1", "chr2"), ignoreStrings)
    }

    /**
     * Test that an empty ignorePatterns list doesn't exclude any contigs in convertGVCFToFasta.
     * This is a regression test for the bug where passing "" would exclude all contigs.
     */
    @Test
    fun testSampleFastaWithEmptyIgnorePatterns() {
        val gvcfFile = "data/LineA.g.vcf"
        val refFile = "data/ref.fa"
        val outputFile = "data/ref_updated.fa"
        val lineAFile = "data/LineA.fa"
        // Simulate what happens when ignoreContig is "" - should result in empty list
        val ignorePatterns = "".split(",").filter { it.isNotBlank() }

        val convertToFasta = ConvertToFasta()
        convertToFasta.convertGVCFToFasta(gvcfFile, refFile, outputFile, ignorePatterns = ignorePatterns)

        // Load in all the fastas to compare the results
        val originalSeq = FastaIO(refFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>
        val updatedSeq = FastaIO(outputFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>
        val lineASeq = FastaIO(lineAFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>

        // Both chromosomes should be present (not excluded)
        assertTrue(updatedSeq.containsKey("1"), "Chrom 1 should be present with empty ignorePatterns")
        assertTrue(updatedSeq.containsKey("2"), "Chrom 2 should be present with empty ignorePatterns")

        // chrom1 should be reference
        assertEquals(originalSeq["1"]!!.seq(), updatedSeq["1"]!!.seq())
        // chrom2 should match LineA's chrom 2
        assertEquals(lineASeq["2"]!!.seq(), updatedSeq["2"]!!.seq())
    }

    @Test
    fun testSampleFasta() {
        val gvcfFile = "data/LineA.g.vcf"
        val refFile = "data/ref.fa"
        val outputFile = "data/ref_updated.fa"
        val lineAFile = "data/LineA.fa"

        val convertToFasta = ConvertToFasta()
        convertToFasta.convertGVCFToFasta(gvcfFile, refFile, outputFile)

        //Load in all the fastas to compare the results
        val originalSeq= FastaIO(refFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>
        val updatedSeq = FastaIO(outputFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>
        val lineASeq = FastaIO(lineAFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>

        //chrom1 should be reference
        assertEquals(originalSeq["1"]!!.seq(), updatedSeq["1"]!!.seq())
        //chrom2 should match LineA's chrom 2
        assertEquals(lineASeq["2"]!!.seq(), updatedSeq["2"]!!.seq())

    }

    @Test
    fun testSampleFastaExcludeChr2() {
        val gvcfFile = "data/LineA.g.vcf"
        val refFile = "data/ref.fa"
        val outputFile = "data/ref_updated.fa"
        val lineAFile = "data/LineA.fa"
        val ignorePatterns = listOf("2")

        val convertToFasta = ConvertToFasta()
        convertToFasta.convertGVCFToFasta(gvcfFile, refFile, outputFile, ignorePatterns = ignorePatterns)

        //Load in all the fastas to compare the results
        val originalSeq= FastaIO(refFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>
        val updatedSeq = FastaIO(outputFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>
        val lineASeq = FastaIO(lineAFile, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>

        //chrom1 should be reference
        assertEquals(originalSeq["1"]!!.seq(), updatedSeq["1"]!!.seq())

        //chrom2 should be excluded
        assertFalse(updatedSeq.containsKey("2"))

    }
}