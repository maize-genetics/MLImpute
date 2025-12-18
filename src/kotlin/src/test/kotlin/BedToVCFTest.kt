import biokotlin.genome.Position
import com.google.common.collect.Range
import com.google.common.collect.RangeMap
import com.google.common.collect.TreeRangeMap
import htsjdk.variant.variantcontext.Allele
import htsjdk.variant.variantcontext.Genotype
import htsjdk.variant.variantcontext.GenotypeBuilder
import htsjdk.variant.variantcontext.VariantContext
import htsjdk.variant.variantcontext.VariantContextBuilder
import htsjdk.variant.vcf.VCFFileReader
import java.io.File
import kotlin.test.DefaultAsserter.assertEquals
import kotlin.test.DefaultAsserter.assertNotNull
import kotlin.test.DefaultAsserter.assertNull
import kotlin.test.DefaultAsserter.assertTrue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.fail


class BedToVCFTest {

    @Test
    fun testProcessBedFiles() {
        val bedToVcf = BedToVcf()

        val bedDir = "data/bedToVCF/"

        val referencePanelVcf = "data/bedToVCF/refPanel.vcf"

        val outFile = "data/bedToVCF/outTestProcess.vcf"

        bedToVcf.processBedFiles(bedDir, referencePanelVcf, outFile)

        //load in the output VCF and check that it has the expected number of variants
        val vcfReader = VCFFileReader(File(outFile), false)
        val variants = vcfReader.iterator().asSequence().toList()
        assertEquals("Expected 6 variants in output VCF, found ${variants.size}", 6,variants.size)

        checkFullVairantList(variants)

        //delete the output file
        File(outFile).delete()
    }

    @Test
    fun testBuildRangeMaps() {
        val bedToVcf = BedToVcf()

        val bedDir = "data/bedToVCF/"

        val rangeMaps = bedToVcf.buildRangeMaps(bedDir)

        //Check that we have the expected number of range maps
        assertEquals("Expected 2 range maps, found ${rangeMaps.size}",2,rangeMaps.size)
        assertTrue("Range map for haploid_1 not found", rangeMaps.containsKey("haploid_1"))
        assertTrue("Range map for diploid_1 not found" ,rangeMaps.containsKey("diploid_1"))

        val truthHaploidMapOfRanges = mapOf(
            Range.closed(Position("chr1", 51), Position("chr1", 200)) to Pair("sample1", "sample1"),
            Range.closed(Position("chr1", 201), Position("chr1", 400)) to Pair("sample1", "sample1"),
            Range.closed(Position("chr1", 401), Position("chr1", 600)) to Pair("sample2", "sample2"),
            Range.closed(Position("chr1", 601), Position("chr1", 700)) to Pair("sample2", "sample2"),
            Range.closed(Position("chr1", 701), Position("chr1", 900)) to Pair("sample1", "sample1")
        )
        //check these ranges
        checkRangeEntries( truthHaploidMapOfRanges, rangeMaps["haploid_1"]!!)

        val truthDiploidMapOfRanges = mapOf(
            Range.closed(Position("chr1", 251), Position("chr1", 350)) to Pair("sample1", "sample1"),
            Range.closed(Position("chr1", 351), Position("chr1", 800)) to Pair("sample1", "sample2"),
            Range.closed(Position("chr1", 801), Position("chr1", 900)) to Pair("sample2", "sample2"),
            Range.closed(Position("chr1", 901), Position("chr1", 1000)) to Pair("sample3", "sample4"),
            Range.closed(Position("chr1", 1001), Position("chr1", 1100)) to Pair("sample3", "sample3"),
            Range.closed(Position("chr1", 1101), Position("chr1", 1200)) to Pair("sample4", "sample4")
        )

        //check these ranges
        checkRangeEntries(truthDiploidMapOfRanges, rangeMaps["diploid_1"]!!)

    }

    @Test
    fun testCreateRangeMapFromBedFile() {
        val bedToVcf = BedToVcf()

        val haploidBedFile = "data/bedToVCF/haploid_1.bed"
        val diploidBedFile = "data/bedToVCF/diploid_1.bed"

        val haploidRangeMap = bedToVcf.createRangeMapFromBedFile(File(haploidBedFile))

        val truthHaploidMapOfRanges = mapOf(
            Range.closed(Position("chr1", 51), Position("chr1", 200)) to Pair("sample1", "sample1"),
            Range.closed(Position("chr1", 201), Position("chr1", 400)) to Pair("sample1", "sample1"),
            Range.closed(Position("chr1", 401), Position("chr1", 600)) to Pair("sample2", "sample2"),
            Range.closed(Position("chr1", 601), Position("chr1", 700)) to Pair("sample2", "sample2"),
            Range.closed(Position("chr1", 701), Position("chr1", 900)) to Pair("sample1", "sample1")
        )
        //check these ranges
        checkRangeEntries( truthHaploidMapOfRanges, haploidRangeMap)



        val diploidRangeMap = bedToVcf.createRangeMapFromBedFile(File(diploidBedFile))

        val truthDiploidMapOfRanges = mapOf(
            Range.closed(Position("chr1", 251), Position("chr1", 350)) to Pair("sample1", "sample1"),
            Range.closed(Position("chr1", 351), Position("chr1", 800)) to Pair("sample1", "sample2"),
            Range.closed(Position("chr1", 801), Position("chr1", 900)) to Pair("sample2", "sample2"),
            Range.closed(Position("chr1", 901), Position("chr1", 1000)) to Pair("sample3", "sample4"),
            Range.closed(Position("chr1", 1001), Position("chr1", 1100)) to Pair("sample3", "sample3"),
            Range.closed(Position("chr1", 1101), Position("chr1", 1200)) to Pair("sample4", "sample4")
        )

        //check these ranges
        checkRangeEntries(truthDiploidMapOfRanges, diploidRangeMap)
    }



    @Test
    fun testBuildNewVcf() {
        val bedToVcf = BedToVcf()

        val inputDir = "data/bedToVCF/"

        val rangeMaps = bedToVcf.buildRangeMaps(inputDir)

        val referencePanelVcf = "data/bedToVCF/refPanel.vcf"

        val outFile = "data/bedToVCF/outTest.vcf"

        bedToVcf.buildNewVcf(referencePanelVcf, rangeMaps, outFile)

        //load in the output VCF and check that it has the expected number of variants
        val vcfReader = VCFFileReader(File(outFile), false)
        val variants = vcfReader.iterator().asSequence().toList()
        assertEquals("Expected 6 variants in output VCF, found ${variants.size}", 6,variants.size)


        checkFullVairantList(variants)
        //delete the output file
        File(outFile).delete()
    }

    private fun checkFullVairantList(variants: List<VariantContext>) {
        //Should match the following:
//        chr1	100	.	A	T	.	.	.	GT	0/0	./.
//        chr1	250	.	A	T	.	.	.	GT	0/0	./.
//        chr1	251	.	A	T	.	.	.	GT	0/0	0/0
//        chr1	500	.	A	T	.	.	.	GT	1/1	0/1
//        chr1	850	.	A	T	.	.	.	GT	0/0	1/1
//        chr1	950	.	A	T	.	.	.	GT	./.	1/0

        val vc1 = variants[0]
        assertEquals(100, vc1.start)
        val gts1 = vc1.genotypes
        assertEquals(2, gts1.size)
        assertEquals("haploid_1", gts1[0].sampleName)
        assertEquals(listOf(Allele.create("A", true), Allele.create("A", true)), gts1[0].alleles)
        assertEquals("diploid_1", gts1[1].sampleName)
        assertEquals(listOf(Allele.NO_CALL, Allele.NO_CALL), gts1[1].alleles)
        val vc2 = variants[1]
        assertEquals(250, vc2.start)
        val gts2 = vc2.genotypes
        assertEquals(2, gts2.size)
        assertEquals("haploid_1", gts2[0].sampleName)
        assertEquals(listOf(Allele.create("A", true), Allele.create("A", true)), gts2[0].alleles)
        assertEquals("diploid_1", gts2[1].sampleName)
        assertEquals(listOf(Allele.NO_CALL, Allele.NO_CALL), gts2[1].alleles)
        val vc3 = variants[2]
        assertEquals(251, vc3.start)
        val gts3 = vc3.genotypes
        assertEquals(2, gts3.size)
        assertEquals("haploid_1", gts3[0].sampleName)
        assertEquals(listOf(Allele.create("A", true), Allele.create("A", true)), gts3[0].alleles)
        assertEquals("diploid_1", gts3[1].sampleName)
        assertEquals(listOf(Allele.create("A", true), Allele.create("A", true)), gts3[1].alleles)
        val vc4 = variants[3]
        assertEquals(500, vc4.start)
        val gts4 = vc4.genotypes
        assertEquals(2, gts4.size)
        assertEquals("haploid_1", gts4[0].sampleName)
        assertEquals(listOf(Allele.create("T", false), Allele.create("T", false)), gts4[0].alleles)
        assertEquals("diploid_1", gts4[1].sampleName)
        assertEquals(listOf(Allele.create("A", true), Allele.create("T", false)), gts4[1].alleles)
        val vc5 = variants[4]
        assertEquals(850, vc5.start)
        val gts5 = vc5.genotypes
        assertEquals(2, gts5.size)
        assertEquals("haploid_1", gts5[0].sampleName)
        assertEquals(listOf(Allele.create("A", true), Allele.create("A", true)), gts5[0].alleles)
        assertEquals("diploid_1", gts5[1].sampleName)
        assertEquals(listOf(Allele.create("T", false), Allele.create("T", false)), gts5[1].alleles)
        val vc6 = variants[5]
        assertEquals(950, vc6.start)
        val gts6 = vc6.genotypes
        assertEquals(2, gts6.size)
        assertEquals("haploid_1", gts6[0].sampleName)
        assertEquals(listOf(Allele.NO_CALL, Allele.NO_CALL), gts6[0].alleles)
        assertEquals("diploid_1", gts6[1].sampleName)
        assertEquals(listOf(Allele.create("T", false), Allele.create("A", true)), gts6[1].alleles)
    }

    @Test
    fun testBuildNewVariantContext() {
        val bedToVcf = BedToVcf()

        val rangeMaps = buildSimpleRangeMap()
        val simpleVariantContext = createSimpleVaraintContext()

        val gameteToAlleleMap = mapOf(
            "sample1" to Allele.create("A", true),
            "sample2" to Allele.create("A", true),
            "sample3" to Allele.create("T", false),
            "sample4" to Allele.NO_CALL
        )

        val newVc = bedToVcf.buildNewVariantContext(rangeMaps, simpleVariantContext, gameteToAlleleMap)

        //check start and chrom
        assertEquals("Start position does not match", simpleVariantContext.start, newVc.start)
        assertEquals("Contig does not match",  simpleVariantContext.contig, newVc.contig)
        //check genotypes
        val genotypes = newVc.genotypes
        checkGenotypes(genotypes)

    }

    @Test
    fun testBuildNewGenotypes() {
        val bedToVcf = BedToVcf()

        val rangeMaps = buildSimpleRangeMap()
        val simpleVariantContext = createSimpleVaraintContext()
        val pos = Position(simpleVariantContext.contig, simpleVariantContext.start)

        val gameteToAlleleMap = mapOf(
            "sample1" to Allele.create("A", true),
            "sample2" to Allele.create("A", true),
            "sample3" to Allele.create("T", false),
            "sample4" to Allele.NO_CALL
        )

        val genotypes = bedToVcf.buildNewGenotypes(rangeMaps, pos, gameteToAlleleMap)
        checkGenotypes(genotypes)

    }

    @Test
    fun testCreateGenericHeader() {
        val bedToVcf = BedToVcf()

        val sampleNames = listOf("sample1", "sample2", "sample3")

        val header = bedToVcf.createGenericHeader(sampleNames,emptySet())
        //Check for some expected header lines
        val expectedLines = listOf(
            "INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">",
            "INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of Samples With Data\">",
            "FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
            "FORMAT=<ID=AD,Number=3,Type=Integer,Description=\"Allelic depths for the ref and alt alleles in the order listed\">"
        )

        //check the sampleNames
        assertEquals(header.sampleNamesInOrder.size, sampleNames.size)
        for (i in sampleNames.indices) {
            assertEquals("Sample name at index $i does not match", header.sampleNamesInOrder[i], sampleNames[i])
        }

        for (line in expectedLines) {
            assertTrue("Header line $line not found", header.idHeaderLines.any { it.toString() == line })
        }
    }

    @Test
    fun testCreateGenericHeaderLineSet() {
        val bedToVcf = BedToVcf()
        val headerLines = bedToVcf.createGenericHeaderLineSet()
        //Check for some expected header lines
        val expectedLines = listOf(
            "INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">",
            "INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of Samples With Data\">",
            "FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
            "FORMAT=<ID=AD,Number=3,Type=Integer,Description=\"Allelic depths for the ref and alt alleles in the order listed\">"
        )
        for (line in expectedLines) {
            assertTrue("Header line $line not found", headerLines.any { it.toString() == line })
        }
    }

    @Test
    fun testBuildGameteToAlleleMap() {
        val bedToVcf = BedToVcf()
        val vc = createSimpleVaraintContext()

        val gameteToAlleleMap = bedToVcf.buildGameteToAlleleMap(vc)
        assertEquals( Allele.create("A", true), gameteToAlleleMap["sample1"])
        assertEquals(Allele.create("A", true), gameteToAlleleMap["sample2"])
        assertEquals(Allele.create("T", false), gameteToAlleleMap["sample3"])
        assertEquals(Allele.NO_CALL, gameteToAlleleMap["sample4"])
        assertFalse(gameteToAlleleMap.containsKey("sample5"))
    }

    private fun createSimpleVaraintContext(): VariantContext {
        //make a simple variant context that we can use to test the allele map creation
        val variantContextBuilder = VariantContextBuilder()
        variantContextBuilder.chr("chr1")
        variantContextBuilder.start(100)
        variantContextBuilder.stop(100)
        variantContextBuilder.alleles(listOf(Allele.create("A", true), Allele.create("T", false)))

        val genotypes = listOf(
            GenotypeBuilder("sample1").alleles(listOf(Allele.create("A", true))).make(),
            GenotypeBuilder("sample2").alleles(listOf(Allele.create("A", true))).make(),
            GenotypeBuilder("sample3").alleles(listOf(Allele.create("T", false))).make(),
            GenotypeBuilder("sample4").alleles(listOf(Allele.NO_CALL)).make(),
            GenotypeBuilder("sample5").make()
        )

        variantContextBuilder.genotypes(genotypes)
        val vc = variantContextBuilder.make()
        return vc
    }

    private fun buildSimpleRangeMap() : Map<String, RangeMap<Position, Pair<String, String>>> {
        val rangeMaps = mutableMapOf<String, RangeMap<Position, Pair<String, String>>>()
        //Build simple range maps for testing

        val test1 = TreeRangeMap.create<Position, Pair<String, String>>()
        test1.put(Range.closed(Position("chr1", 50), Position("chr1", 150)), Pair("sample1", "sample1"))
        rangeMaps["test_sample1"] = test1

        val test2 = TreeRangeMap.create<Position, Pair<String, String>>()
        test2.put(Range.closed(Position("chr1", 75), Position("chr1", 125)), Pair("sample1", "sample3"))
        rangeMaps["test_sample2"] = test2

        val test3 = TreeRangeMap.create<Position, Pair<String, String>>()
        test3.put(Range.closed(Position("chr1", 90), Position("chr1", 110)), Pair("sample3", "sample3"))
        rangeMaps["test_sample3"] = test3

        val test4 = TreeRangeMap.create<Position, Pair<String, String>>()
        test4.put(Range.closed(Position("chr1", 200), Position("chr1", 300)), Pair("sample2", "sample2"))
        rangeMaps["test_sample4"] = test4

        //test no calls
        val test5 = TreeRangeMap.create<Position, Pair<String, String>>()
        test5.put(Range.closed(Position("chr1", 50), Position("chr1", 150)), Pair("sample4", "sample4"))
        rangeMaps["test_sample5"] = test5

        //Test no calls gamete not in the vcf
        val test6 = TreeRangeMap.create<Position, Pair<String, String>>()
        test6.put(Range.closed(Position("chr1", 50), Position("chr1", 150)), Pair("sample10", "sample10"))
        rangeMaps["test_sample6"] = test6


        return rangeMaps
    }

    private fun checkGenotypes(genotypes: MutableList<Genotype>) {
        //Check that we have the expected genotypes
        assertEquals("Expected 5 genotypes, found ${genotypes.size}", 5, genotypes.size)
        val genotypeMap = genotypes.associateBy { it.sampleName }
        val g1 = genotypeMap["test_sample1"]
        val g2 = genotypeMap["test_sample2"]
        val g3 = genotypeMap["test_sample3"]
        val g4 = genotypeMap["test_sample4"]
        val g5 = genotypeMap["test_sample5"]
        val g6 = genotypeMap["test_sample6"]
        assertNotNull("Genotype for test_sample1 not found", g1)
        assertNotNull("Genotype for test_sample2 not found", g2)
        assertNotNull("Genotype for test_sample3 not found", g3)
        assertNull("Genotype for test_sample4 was found", g4)
        assertNotNull("Genotype for test_sample5 not found", g5)
        assertNotNull("Genotype for test_sample6 not found", g6)

        assertEquals("Unexpected first allele for test_sample1",Allele.create("A", true),g1!!.alleles[0] )
        assertEquals("Unexpected second allele for test_sample1", Allele.create("A", true),g1.alleles[1] )
        assertEquals("Unexpected first allele for test_sample2",Allele.create("A", true), g2!!.alleles[0] )
        assertEquals("Unexpected second allele for test_sample2", Allele.create("T", false),g2.alleles[1] )
        assertEquals("Unexpected first allele for test_sample3", Allele.create("T", false), g3!!.alleles[0])
        assertEquals("Unexpected second allele for test_sample3", Allele.create("T", false), g3.alleles[1] )

        assertEquals("Unexpected first allele for test_sample5", Allele.NO_CALL, g5!!.alleles[0] )
        assertEquals("Unexpected second allele for test_sample5", Allele.NO_CALL, g5.alleles[1])

        //g6 should be no call as the gamete is not in the map
        assertEquals("Unexpected first allele for test_sample6", Allele.NO_CALL, g6!!.alleles[0] )
        assertEquals("Unexpected second allele for test_sample6", Allele.NO_CALL, g6.alleles[1] )
    }

    private fun checkRangeEntries(
        truthHaploidMapOfRanges: Map<Range<Position>, Pair<String, String>>,
        inputRangeMap: RangeMap<Position, Pair<String, String>>
    ) {
        for (entry in inputRangeMap.asMapOfRanges()) {
            val expectedValue = truthHaploidMapOfRanges[entry.key]
            assertTrue("Unexpected range found: ${entry.key}", expectedValue != null)
            assertEquals("For range ${entry.key}, expected value $expectedValue but found ${entry.value}", expectedValue, entry.value)
        }

        //Check the inverse
        for (entry in truthHaploidMapOfRanges) {
            val foundValue = inputRangeMap.get(entry.key.lowerEndpoint())
            assertTrue("Expected range ${entry.key} not found in range map", foundValue != null)
            assertEquals("For range ${entry.key}, expected value ${entry.value} but found $foundValue" ,foundValue, entry.value)
        }
    }
}