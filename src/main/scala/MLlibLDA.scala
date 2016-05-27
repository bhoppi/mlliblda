import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

import scala.annotation.tailrec


object MLlibLDA {
  type OptionMap = Map[String, String]

  def main(args: Array[String]) {
    val options = parseArgs(args)
    val appStartedTime = System.nanoTime

    val numTopics = options("numtopics").toInt
    val alpha = options("alpha").toDouble
    val beta = options("beta").toDouble
    val totalIter = options("totaliter").toInt
    val numPartitions = options("numpartitions").toInt
    val numVocab = options("numvocab").toInt
    assert(numTopics > 0, "numTopics must be greater than 0")
    assert(alpha > 0.0)
    assert(beta > 0.0)
    assert(totalIter > 0, "totalIter must be greater than 0")
    assert(numPartitions > 0, "numPartitions must be greater than 0")

    val inputPath = options("inputpath")
    val outputPath = options("outputpath")
    val checkpointPath = outputPath + ".checkpoint"

    val conf = new SparkConf()
    val outPath = new Path(outputPath)
    val fs = getFileSystem(conf, outPath)
    if (fs.exists(outPath)) {
      println(s"Error: output path $outputPath already exists.")
      System.exit(2)
    }
    fs.delete(new Path(checkpointPath), true)

    val sc = new SparkContext(conf)
    try {
      sc.setCheckpointDir(checkpointPath)
      println("start LDA training")
      println(s"appId: ${sc.applicationId}")
      println(s"numTopics = $numTopics, totalIteration = $totalIter")
      println(s"alpha = $alpha, beta = $beta")
      println(s"inputDataPath = $inputPath")
      println(s"outputPath = $outputPath")

      val rawDocs = sc.textFile(inputPath, numPartitions)
      val corpus = rawDocs.map { line =>
        val tokens = line.split(raw"\t|\s+").view
        val docId = tokens.head.toLong
        val bow = Vectors.sparse(numVocab, tokens.tail.map { field =>
          val pairs = field.split(":")
          val termId = pairs(0).toInt
          val termCnt = if (pairs.length > 1) pairs(1).toInt else 1
          (termId, termCnt.toDouble)
        })
        (docId, bow)
      }.cache()
      val lda = new LDA()
        .setAlpha(alpha + 1)
        .setBeta(beta + 1)
        .setK(numTopics)
        .setMaxIterations(totalIter)
      val ldaModel = lda.run(corpus)

      ldaModel.save(sc, outputPath)
    } finally {
      sc.stop()
      fs.deleteOnExit(new Path(checkpointPath))
      val appEndedTime = System.nanoTime
      println(s"Total time consumed: ${(appEndedTime - appStartedTime) / 1e9} seconds")
      fs.close()
    }
  }

  def getFileSystem(conf: SparkConf, path: Path): FileSystem = {
    val hadoopConf = SparkHadoopUtil.get.newConfiguration(conf)
    if (sys.env.contains("HADOOP_CONF_DIR") || sys.env.contains("YARN_CONF_DIR")) {
      val hdfsConfPath = if (sys.env.get("HADOOP_CONF_DIR").isDefined) {
        sys.env.get("HADOOP_CONF_DIR").get + "/core-site.xml"
      } else {
        sys.env.get("YARN_CONF_DIR").get + "/core-site.xml"
      }
      hadoopConf.addResource(new Path(hdfsConfPath))
    }
    path.getFileSystem(hadoopConf)
  }

  def parseArgs(args: Array[String]): OptionMap = {
    val usage = "Usage: MLlibLDA <Args> [Options] <Input path> <Output path>\n" +
      "  Args: -numTopics=<Int> -alpha=<Double> -beta=<Double>\n" +
      "        -totalIter=<Int> -numPartitions=<Int> -numVocab=<Int>"
    if (args.length < 8) {
      println(usage)
      System.exit(1)
    }
    val arglist = args.toList
    @tailrec def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      def isSwitch(s: String) = s(0) == '-'
      list match {
        case Nil => map
        case head :: Nil if !isSwitch(head) =>
          nextOption(map ++ Map("outputpath" -> head), Nil)
        case head :: tail if !isSwitch(head) =>
          nextOption(map ++ Map("inputpath" -> head), tail)
        case head :: tail if isSwitch(head) =>
          var kv = head.toLowerCase.split("=", 2)
          if (kv.length == 1) {
            kv = head.toLowerCase.split(":", 2)
          }
          if (kv.length == 1) {
            println(s"Error: wrong command line format: $head")
            System.exit(1)
          }
          nextOption(map ++ Map(kv(0).substring(1) -> kv(1)), tail)
        case _ =>
          println(usage)
          System.exit(1)
          null.asInstanceOf[OptionMap]
      }
    }
    nextOption(Map(), arglist)
  }
}
