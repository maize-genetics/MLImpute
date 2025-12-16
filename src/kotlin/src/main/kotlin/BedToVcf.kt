import biokotlin.genome.Position
import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.required
import com.google.common.collect.RangeMap
import com.google.common.collect.TreeRangeMap
import htsjdk.variant.variantcontext.Allele
import htsjdk.variant.variantcontext.Genotype
import htsjdk.variant.variantcontext.GenotypeBuilder
import htsjdk.variant.variantcontext.GenotypesContext
import htsjdk.variant.variantcontext.VariantContext
import htsjdk.variant.variantcontext.VariantContextBuilder
import htsjdk.variant.variantcontext.writer.Options
import htsjdk.variant.variantcontext.writer.VariantContextWriterBuilder
import htsjdk.variant.vcf.VCFFileReader
import htsjdk.variant.vcf.VCFFormatHeaderLine
import htsjdk.variant.vcf.VCFHeader
import htsjdk.variant.vcf.VCFHeaderLine
import htsjdk.variant.vcf.VCFHeaderLineCount
import htsjdk.variant.vcf.VCFHeaderLineType
import htsjdk.variant.vcf.VCFInfoHeaderLine
import java.io.File
import java.util.HashSet

class BedToVcf : CliktCommand(help = "Convert a set of imputed BED files to a single VCF file") {

    val bedDir: String by option(help="BED directory").required()
    val referencePanelVcf : String by option(help="Reference panel VCF file").required()
    val outFile: String by option(help="Output file").required()


    override fun run() {
        processBedFiles(bedDir, referencePanelVcf, outFile)
    }

    fun processBedFiles(bedDir: String, referencePanelVcf: String, outFile: String) {
        val rangeMaps = buildRangeMaps(bedDir)

        buildNewVcf(referencePanelVcf, rangeMaps, outFile)
    }

    fun buildRangeMaps(bedDir: String): Map<String, RangeMap<Position,Pair<String,String>>> {
        val ranges = mutableMapOf<String, RangeMap<Position,Pair<String,String>>>()
        //Loop through each bed file in the directory
        File(bedDir).listFiles { file -> file.extension == "bed" }?.forEach { bedFile ->
            val sampleId = bedFile.nameWithoutExtension
            val rangeMap = createRangeMapFromBedFile(bedFile)
            ranges[sampleId] = rangeMap
        }
        return ranges
    }

    fun createRangeMapFromBedFile(bedFile: File): RangeMap<Position, Pair<String, String>> {
        // Implement logic to read BED file and create RangeMap
        val rangeMap = TreeRangeMap.create<Position, Pair<String, String>>()
        //Loop through the BED file
        bedFile.forEachLine { line ->
            val parts = line.split("\t")
            if (parts.size >= 4) {
                val chrom = parts[0]
                val start = parts[1].toInt()
                val end = parts[2].toInt()
                val genotype1 = parts[3]
                val genotype2 = if (parts.size >=5) parts[4] else genotype1
                val posStart = Position(chrom, start+1 ) // BED is 0-based, Position is 1-based
                val posEnd = Position(chrom, end)
                rangeMap.put(com.google.common.collect.Range.closedOpen(posStart, posEnd), Pair(genotype1, genotype2))
            }
        }

        return rangeMap
    }

    fun buildNewVcf(referencePanelVcf: String, rangeMaps: Map<String, RangeMap<Position,Pair<String,String>>>, outFile: String) {
        // Implement logic to read reference VCF and write new VCF using rangeMaps
        // This is a placeholder for the actual implementation
        println("Building new VCF from $referencePanelVcf using BED data and writing to $outFile")
        //Build a VCF writer
        VariantContextWriterBuilder()
            .unsetOption(Options.INDEX_ON_THE_FLY)
            .setOutputFile(File(outFile))
            .setOutputFileType(VariantContextWriterBuilder.OutputType.VCF)
            .setOption(Options.ALLOW_MISSING_FIELDS_IN_HEADER)
            .build().use { writer ->
                writer.writeHeader(createGenericHeader(rangeMaps.keys.toList(), emptySet()))


                //Open up the reference VCF and loop through each record
                val variantReader = VCFFileReader(File(referencePanelVcf), false)

                val iterator = variantReader.iterator()

                while(iterator.hasNext()) {
                    val vc = iterator.next()

                    val gameteToAlleleMap = buildGameteToAlleleMap(vc)

                    val newVc = buildNewVariantContext(rangeMaps, vc, gameteToAlleleMap)

                    //Write the new variant context to the VCF
                    writer.add(newVc)

                }

            }
    }

    fun buildNewVariantContext(
        rangeMaps: Map<String, RangeMap<Position, Pair<String, String>>>,
        vc: VariantContext,
        gameteToAlleleMap: Map<String, Allele>
    ): VariantContext {
        val genotypeList = buildNewGenotypes(rangeMaps, vc, gameteToAlleleMap)

        //Build a new variant context with the new genotype
        val newVc = VariantContextBuilder(vc)
            .genotypes(genotypeList)
            .make()
        return newVc
    }

    fun buildNewGenotypes(
        rangeMaps: Map<String, RangeMap<Position, Pair<String, String>>>,
        vc: VariantContext,
        gameteToAlleleMap: Map<String, Allele>
    ): MutableList<Genotype> {
        val genotypeList = mutableListOf<Genotype>()

        for (sampleName in rangeMaps.keys) {
            val rangeMap = rangeMaps[sampleName]!!
            val pos = Position(vc.contig, vc.start)
            val genotypePair = rangeMap.get(pos)
            if (genotypePair != null) {
                val gamete1 = genotypePair.first
                val gamete2 = genotypePair.second

                val allele1 = gameteToAlleleMap[gamete1] ?: Allele.NO_CALL
                val allele2 = gameteToAlleleMap[gamete2] ?: Allele.NO_CALL
                //Build a new genotype object
                val newGenotype = GenotypeBuilder(sampleName)
                    .alleles(listOf(allele1, allele2))
                    .make()

                genotypeList.add(newGenotype)
            }
        }
        return genotypeList
    }

    fun createGenericHeader(taxaNames: List<String>, altLines:Set<VCFHeaderLine>): VCFHeader {
        val headerLines = createGenericHeaderLineSet() as MutableSet<VCFHeaderLine>
        headerLines.addAll(altLines)
        return VCFHeader(headerLines, taxaNames)
    }

    fun createGenericHeaderLineSet(): Set<VCFHeaderLine> {
        val headerLines: MutableSet<VCFHeaderLine> = HashSet()
        headerLines.add(VCFFormatHeaderLine("AD", 3, VCFHeaderLineType.Integer, "Allelic depths for the ref and alt alleles in the order listed"))
        headerLines.add(
            VCFFormatHeaderLine("DP", 1, VCFHeaderLineType.Integer, "Read Depth (only filtered reads used for calling)")
        )
        headerLines.add(VCFFormatHeaderLine("GQ", 1, VCFHeaderLineType.Integer, "Genotype Quality"))
        headerLines.add(VCFFormatHeaderLine("GT", 1, VCFHeaderLineType.String, "Genotype"))
        headerLines.add(
            VCFFormatHeaderLine("PL", VCFHeaderLineCount.G, VCFHeaderLineType.Integer, "Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification")
        )
        headerLines.add(VCFInfoHeaderLine("DP", 1, VCFHeaderLineType.Integer, "Total Depth"))
        headerLines.add(VCFInfoHeaderLine("NS", 1, VCFHeaderLineType.Integer, "Number of Samples With Data"))
        headerLines.add(VCFInfoHeaderLine("AF", 3, VCFHeaderLineType.Integer, "Allele Frequency"))
        headerLines.add(VCFInfoHeaderLine("END", 1, VCFHeaderLineType.Integer, "Stop position of the interval"))
        // These last 4 are needed for assembly g/hvcfs, but not for reference.  I am keeping them in as header
        // lines but they will only be added to the data lines if they are present in the VariantContext.
        headerLines.add(VCFInfoHeaderLine("ASM_Chr", 1, VCFHeaderLineType.String, "Assembly chromosome"))
        headerLines.add(VCFInfoHeaderLine("ASM_Start", 1, VCFHeaderLineType.Integer, "Assembly start position"))
        headerLines.add(VCFInfoHeaderLine("ASM_End", 1, VCFHeaderLineType.Integer, "Assembly end position"))
        headerLines.add(VCFInfoHeaderLine("ASM_Strand", 1, VCFHeaderLineType.String, "Assembly strand"))
        return headerLines
    }

    fun buildGameteToAlleleMap(vc: VariantContext): Map<String, Allele> {
        //Loop through each sample in the variant context
        return vc.genotypes.map { genotype ->
            val sampleName = genotype.sampleName
            val alleles = genotype.alleles
            if(alleles.isNotEmpty()) {
                Pair(sampleName, alleles[0])
            }
            else {
                Pair("", Allele.NO_CALL)
            }
        }.filter { it.first != "" }.toMap()
    }
}