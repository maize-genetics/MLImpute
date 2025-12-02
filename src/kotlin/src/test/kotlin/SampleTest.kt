import biokotlin.seq.NucSeqRecord
import biokotlin.seqIO.FastaIO
import biokotlin.seqIO.SeqType
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test


class SampleTest {

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