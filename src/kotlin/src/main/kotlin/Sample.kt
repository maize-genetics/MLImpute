import biokotlin.seq.NucSeqRecord
import biokotlin.seqIO.FastaIO
import biokotlin.seqIO.SeqType
import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.options.required
import com.github.ajalt.clikt.parameters.types.boolean
import com.github.ajalt.clikt.parameters.types.enum
import com.github.ajalt.clikt.parameters.types.int
import htsjdk.variant.variantcontext.writer.Options
import htsjdk.variant.variantcontext.writer.VariantContextWriterBuilder
import htsjdk.variant.vcf.VCFContigHeaderLine
import htsjdk.variant.vcf.VCFFileReader
import htsjdk.variant.vcf.VCFHeader
import java.io.File
import kotlin.io.path.*
import kotlin.random.Random

enum class MissingGT {asN, asRef, asNone}
/**
 * Simple class to place newline characters at fixed intervals while writing to output
 */
class FastaLineWrapper(val wrapSize: Int = 60){
    var counter = 0

    /**
     * Resets the character counter
     */
    fun reset() {
        counter = 0
    }

    /**
     * Takes an input string and places newline characters at fixed intervals.
     * This adds with previous strings, so that a single line will never exceed wrapSize.
     * Returns the string with added newline characters.
     */
    fun wrapLine(inLine: String): String {
        val leftToNewLine = wrapSize - counter

        if(inLine.length < leftToNewLine) {

            counter += inLine.length
            return inLine
        } else if(inLine.length == leftToNewLine) {
            counter = 0

            return "$inLine\n"
        } else { // string must be split over multiple lines
            val firstLine = inLine.substring(0, leftToNewLine)

            val lineChunks = inLine.substring(leftToNewLine).chunked(wrapSize)

            if (lineChunks.last().length == wrapSize) {
                counter = 0

                return "$firstLine\n${lineChunks.joinToString("\n")}\n"
            } else {
                counter = lineChunks.last().length
                return "$firstLine\n${lineChunks.joinToString("\n")}"
            }
        }
    }
}

class ConvertToFasta: CliktCommand(help="generate fasta from GVCF") {
    val gvcfFile: String by option(help="gvcf file").required()
    val outFile: String by option(help="out fasta").required()
    val fastaFile: String by option(help="ref fasta").required()
    val missingRecordsAs by option(help="if a position is missing a gvcf record (variant or ref block), fill " +
            "with N's (asN), reference (asRef) or omit sequence (asNone). Default asRef").enum<MissingGT>().default(MissingGT.asRef)
    val missingGenotypeAs by option(help="if the sample has a missing genotype (.), fill the position with N's (asN)," +
            "reference (asRef), or omit sequence (asNone)").enum<MissingGT>().default(MissingGT.asN)

    override fun commandHelpEpilog(context: Context): String {
        return "Constructs a fasta file based on the variants listed in the given GVCF file. "
    }

    override fun run() {
        convertGVCFToFasta(gvcfFile, fastaFile, outFile, missingRecordsAs = missingRecordsAs, missingGenotypeAs = missingGenotypeAs)
    }

    /** Function to convert a genotype in a GVCF file to a fasta sequence.
     * All records must use non-symbloic alleles (except <NON_REF>), and duplicated positions are not allowed.
     * Supports multisample VCFs.
     * Parameters:
     *  gvcfFile: path to the GVCF file to use. The script accepts VCF file format, but GVCFs should be used to ensure that
     *      all variants are accounted for
     *  refFasta: path to the reference FASTA file
     *  outFile: path to the output FASTA file
     *  sampleName: optional, the sample name to use in a multisample VCF. Defaults to the first sample listed
     *  missingAsRef: optional, default true. If true, treat missing positions as reference blocks. If false, omit missing positions.
     *  alleleIdx: optional. In a diploid or polyploid, the index of the allele to use. Defaults to 0.
     */
    fun convertGVCFToFasta(gvcfFile: String, refFasta: String, outFile: String, sampleName: String? = null,
                           missingRecordsAs: MissingGT = MissingGT.asRef, missingGenotypeAs: MissingGT = MissingGT.asN,
                           alleleIdx: Int = 0){

        // stream directly to output to save on RAM
        File(outFile).bufferedWriter().use{writer ->

            // read  files
            val reader = VCFFileReader(File(gvcfFile), false)
            val iterator = reader.iterator()

            val lineWrapper = FastaLineWrapper()

            val fasta = FastaIO(refFasta, SeqType.nucleotide).readAll() as Map<String, NucSeqRecord>

            // if sampleName was not specified, use the first sample
            val sampleNames = reader.header.genotypeSamples
            val name = sampleName ?: sampleNames[0]

            var previousRecordEnd = 0
            var previousRecordStart = -1
            var previousChrom = "null"
            var fastaSeq = "" // contigs should be continuous, so use this to save string conversion time each record

            val seenChroms: MutableList<String> = mutableListOf()

            // process each VCF record in sequence
            for (record in iterator) {
                val chrom = record.contig

                // case: new contig encountered
                if(previousChrom != chrom) {

                    // fill in the last ref block if it wasn't explicitly recorded
                    if(previousChrom != "null") {
                        if (previousRecordEnd < fasta[previousChrom]!!.size()) {
                            if (missingRecordsAs == MissingGT.asN) {
                                val seq0 = "N".repeat(fasta[previousChrom]!!.size()-previousRecordEnd)
                                writer.write(lineWrapper.wrapLine(seq0))
                            } else if (missingRecordsAs == MissingGT.asRef) {
                                val seq0 = fastaSeq.substring(previousRecordEnd, fasta[previousChrom]!!.size())
                                writer.write(lineWrapper.wrapLine(seq0))
                            }
                        }
                    }

                    // write fasta info line
                    if(previousChrom == "null") {
                        writer.write(">$chrom\n")
                    } else {
                        writer.write("\n>$chrom\n")
                    }

                    // ordering issue
                    check(chrom !in seenChroms) { "Chromosomes are not contiguous!" }

                    seenChroms.add(chrom)
                    lineWrapper.reset()

                    previousChrom = chrom
                    previousRecordEnd = 0
                    previousRecordStart = -1
                    fastaSeq = fasta[chrom]!!.seq()
                }

                check(previousRecordStart < record.start) { "Record positions must be strictly increasing! ${record.start} $previousRecordStart"}

                // if there is a gap between the previous record and the current, treat according to missingAsRef flag
                if (previousRecordEnd < (record.start - 1)) {
                    if (missingRecordsAs == MissingGT.asN) {
                        val seq0 = "N".repeat(record.start - previousRecordEnd - 1)
                        writer.write(lineWrapper.wrapLine(seq0))
                    } else if (missingRecordsAs == MissingGT.asRef) {
                        val seq0 = fastaSeq.substring(previousRecordEnd, record.start-1)
                        writer.write(lineWrapper.wrapLine(seq0))
                    }
                }

                // get the specific variant to write
                val genotype = record.getGenotype(name)
                val allele = genotype.getAllele(alleleIdx)

                // write the variant
                val seq = if(allele.isReference){
                    fastaSeq.substring(record.start-1, record.end)
                } else {
                    check(!allele.isSymbolic) { "GVCF may not use symbolic alleles, except for <NON_REF>"}
                    if(allele.isNoCall) {
                        if(missingGenotypeAs == MissingGT.asN) {
                            "N".repeat(record.lengthOnReference)
                        } else if (missingGenotypeAs == MissingGT.asRef) {
                            fastaSeq.substring(record.start-1, record.end)
                        } else {
                            ""
                        }
                    } else {
                        allele.baseString
                    }
                }
                writer.write(lineWrapper.wrapLine(seq))

                previousRecordEnd = record.end
                previousRecordStart = record.start
            }

            // fill in the last ref block if it wasn't explicitly recorded
            if(previousChrom != "null") {
                if (previousRecordEnd < fasta[previousChrom]!!.size()) {
                    if (missingRecordsAs == MissingGT.asN) {
                        val seq0 = "N".repeat(fasta[previousChrom]!!.size()-previousRecordEnd)
                        writer.write(lineWrapper.wrapLine(seq0))
                    } else if (missingRecordsAs == MissingGT.asRef) {
                        val seq0 = fastaSeq.substring(previousRecordEnd, fasta[previousChrom]!!.size())
                        writer.write(lineWrapper.wrapLine(seq0))
                    }
                }
            }

            reader.close()

            //Need to go through the list of seen chroms and make sure all chroms in the fasta were seen
            val seenChromsSet = seenChroms.toSet()
            fasta.keys.filter { !seenChromsSet.contains(it) }.forEach { key ->
                //write out the entire chrom as missing
                writer.write("\n>$key\n")
                lineWrapper.reset()

                val seq0 = fasta[key]!!.seq()
                writer.write(lineWrapper.wrapLine(seq0))

            }

        }

    }


}

class DownsampleGvcf: CliktCommand(help="Sample variants from GVCF"){
    val gvcfDir: String by option(help="gvcf directory").required()
    val outDir: String by option(help="out gvcf directory").required()
    val ignoreContig: String by option(help="comma-separated list of string patterns to ignore").default("")
    val rates: String by option(help="comma-separated list of downsampling rates to use for each chromosome")
        .default("0.01,0.05,0.1,0.15,0.2,0.3,0.35,0.4,0.45,0.49")
    val seed: Int? by option(help="random seed").int()
    val keepRef: Boolean by option(help="keep ref blocks").boolean().default(true)
    val minRefBlockSize: Int by option(help="minimum reference block size to sample").int().default(20)

    override fun commandHelpEpilog(context: Context): String {
        return "Sample variants from a GVCF file at a fixed rate per chromosome"
    }
    override fun run() {
        val ignoreStrings = ignoreContig.split(",")

        Path(gvcfDir).forEachDirectoryEntry { filePath ->
            if(filePath.extension == "gvcf" || filePath.extension == "vcf") {
                val inFile = filePath.toString()
                val outFile =  "$outDir/${filePath.fileName.nameWithoutExtension}_subsampled.gvcf"


                val rateList = rates.split(",").map{it.toDouble()}

                downsample(inFile, outFile, rateList, ignorePatterns = ignoreStrings,
                    keepRef=keepRef, randomSeed = seed, minRefBlockSize = minRefBlockSize)

            }
        }
    }

    fun downsample(gvcfFile: String, outFile: String, sampleRates: List<Double> = listOf(), sampleName: String? = null, ignorePatterns: List<String> = listOf(),
                   keepRef: Boolean = true, randomSeed: Int? = null, minRefBlockSize: Int = 20, randomizeSampleRates: Boolean = true) {
        val rand = if(randomSeed != null) {
            Random(randomSeed)
        } else {
            Random
        }

        val reader = VCFFileReader(File(gvcfFile), false)
        val iterator = reader.iterator()


        var sampleRates = if(randomizeSampleRates) {
            sampleRates.shuffled(rand)
        } else {
            sampleRates
        }

        if(sampleRates.size == 0) { // randomly generate sample rates
            sampleRates = (0..reader.header.contigLines.size).map{rand.nextInt(1, 49)  / 100.0}
        } else {
            assert(reader.header.contigLines.size == sampleRates.size) {"Number of chosen sample rates does not match contig number"}
            assert(sampleRates.all{it < 0.5 && it > 0}) {"Sample rates must be between 0 and 50%, exclusive"}
        }

        val sampleNames = reader.header.genotypeSamples
        val name = sampleName ?: sampleNames[0]

        var idCounter = -1

        val newMetaSet = reader.header.metaDataInInputOrder.map{
            if(it is VCFContigHeaderLine) {
                if (ignorePatterns.isNotEmpty()) {
                    if(ignorePatterns.any{pattern -> it.id.contains(pattern)}) {
                        null
                    } else {
                        idCounter += 1
                        val newMap = it.genericFields.toMutableMap()
                        newMap["sample_rate"] = sampleRates[idCounter].toString()
                        VCFContigHeaderLine(newMap, idCounter)
                    }
                } else {
                    idCounter += 1
                    val newMap = it.genericFields.toMutableMap()
                    newMap["sample_rate"] = sampleRates[idCounter].toString()
                    VCFContigHeaderLine(newMap, idCounter)
                }
            } else {
                it
            }
        }.filterNotNull().toSet()

        val header = VCFHeader(newMetaSet, reader.header.genotypeSamples)

        val contigLines = header.contigLines.sortedBy{it.contigIndex}.map{it.id}

        val writer = VariantContextWriterBuilder()
            .unsetOption(Options.INDEX_ON_THE_FLY)
            .setOutputFile(File(outFile))
            .setOutputFileType(VariantContextWriterBuilder.OutputType.VCF)
            .setOption(Options.ALLOW_MISSING_FIELDS_IN_HEADER)
            .build()

        writer.writeHeader(header)

        val blockSizeFileName = "${outFile.split(".").dropLast(1).joinToString(".")}_block_sizes.tsv"

        var sampleRate = 0.0
        var lastContig = ""
        var keepBlock = true
        var canChangeBlock = true

        var lastBlockChange = 0


        File(blockSizeFileName).bufferedWriter().use{sizeWriter ->
            sizeWriter.write("chrom\tblockSize\tblockID\n")
            for(record in iterator) {
                if(record.contig != lastContig) {
                    if(!contigLines.contains(record.contig)) {
                        continue
                    }
                    sampleRate = sampleRates[contigLines.indexOf(record.contig)]
                    lastContig = record.contig
                    canChangeBlock = true

                    lastBlockChange = 0

                }

                val genotype = record.getGenotype(name)
                val allele = genotype.getAllele(0)

                if(allele.isReference) { // keep reference alleles if flag is set
                    if(keepRef) {
                        writer.add(record)
                    }
                    if(record.lengthOnReference >= minRefBlockSize) {
                        canChangeBlock = true
                    }
                } else {
                    if(canChangeBlock) {
                        if(rand.nextDouble() < sampleRate) {
                            if(!keepBlock) {
                                sizeWriter.write("${record.contig}\t${record.start - lastBlockChange}\tref\n")
                                lastBlockChange = record.start
                            }
                            keepBlock = true
                        } else {
                            if(keepBlock) {
                                sizeWriter.write("${record.contig}\t${record.start - lastBlockChange}\tquery\n")
                                lastBlockChange = record.start
                            }
                            keepBlock = false
                        }
                        canChangeBlock = false
                    }

                    if(keepBlock) {
                        writer.add(record)
                    }
                }
            }
        }


        writer.close()
    }
}

class Sample: CliktCommand() {
    override fun run() = Unit
}


fun main(args: Array<String>) {
    Sample().subcommands(DownsampleGvcf(), ConvertToFasta()).main(args)
}