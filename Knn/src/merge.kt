import java.io.File
import java.lang.Integer.max
import java.lang.Exception
import java.lang.Math.abs
import java.lang.Math.sqrt

typealias Vector = ArrayList<Long>
typealias Prediction = List<Pair<Int, Double>>

class KNNClassifier(
        private val neighbors: Int = 5,
        private val window_size: Double = -1.0,
        private val kernel: String = "triangular",
        private val metric: String = "euclidean"
) {
    private lateinit var data: List<Vector>
    private lateinit var target: List<Int>

    fun fit(X: List<Vector>, y: List<Int>) {
        data = X
        target = y
    }

    fun predict(X: List<Vector>): List<Prediction> {
        return X.map { x: Vector ->
            val temp = data.asSequence()
                    .mapIndexed { ind, it -> Pair(ind, dist(x, it)) }
                    .sortedBy { a -> a.second }
                    .take(neighbors)
            val window = if (window_size > 0) window_size else temp.last().second
            temp
                    .map { Pair(it.first, weight(it.second / window)) }
                    .sortedByDescending { a -> a.second }
                    .toList()
        }
    }

    private fun dist(a: Vector, b: Vector): Double {
        return when (metric) {
            "euclidean" -> sqrt((a zip b).fold(0.0) { acc, (a, b) ->
                acc + (a - b) * (a - b)
            })
            "manhattan" -> (a zip b).fold(0.0) { acc, (a, b) ->
                acc + abs(a - b)
            }
            else -> throw Exception("Bad metric = $metric")
        }
    }

    private fun weight(dist: Double): Double {
        if (dist in -1.0..1.0) {
            return when (kernel) {
                "uniform" -> 0.5
                "triangular" -> 1 - abs(dist)
                "parabolic" -> 0.75 * (1 - dist * dist)
                "biweight" -> 0.9375 * (1 - dist * dist) * (1 - dist * dist)
                "triweight" -> 1.09375 * (1 - dist * dist) * (1 - dist * dist) * (1 - dist * dist)
                "tricube" -> {
                    val a = abs(dist)
                    70 / 81 * (1 - a * a * a) * (1 - a * a * a) * (1 - a * a * a)
                }
                else -> throw Exception("Bad kernel = $kernel")
            }
        }
        return 0.0
    }
}

fun validate(clf: KNNClassifier, X: List<Vector>, y: List<Int>, cv: Int = 10): Double {
    val classCount = y.max()!!
    val classes = Array<ArrayList<Vector>>(classCount + 1) { ArrayList() }
    val partsX = ArrayList<ArrayList<Vector>>()
    val partsY = ArrayList<ArrayList<Int>>()
    for (i in (0 until cv)) {
        partsX.add(ArrayList())
        partsY.add(ArrayList())
    }
    for ((x, c) in (X zip y)) {
        classes[c].add(x)
    }
    var pos = 0
    for (i in (1..classCount)) {
        for (x in classes[i]) {
            partsX[pos].add(x)
            partsY[pos].add(i)
            pos++
            pos %= cv
        }
    }

    return (0 until cv).sumByDouble { i ->
        val testX = partsX[i]
        val testY = partsY[i]
        val trainX = partsX.filterIndexed { index, _ -> index != i }.flatten()
        val trainY = partsY.filterIndexed { index, _ -> index != i }.flatten()
        clf.fit(trainX, trainY)
        fScore(clf.predict(testX).map { trainY[it.first().first] }, testY)
    } / cv
}

fun fScore(expected: List<Int>, predicted: List<Int>): Double {
    if (expected.size != predicted.size) throw Exception("Predicted size = ${predicted.size}, Expected size = ${expected.size}")
    val classes = max(expected.max()!!, predicted.max()!!) + 1
    val tp = IntArray(classes)
    val fp = IntArray(classes)
    val fn = IntArray(classes)
    val count = IntArray(classes)
    var sum = 0

    (expected zip predicted).forEach { (expected, predicted) ->
        if (expected == predicted) {
            tp[expected] += 1
        } else {
            fp[expected] += 1
            fn[predicted] += 1
        }
        count[expected] += 1
        sum += 1
    }

    var precision = 0.0
    var recall = 0.0
    for (i in (1 until classes)) {
        val t = tp[i] + fn[i]
        if (tp[i] + fp[i] != 0) precision += t * tp[i].toDouble() / (tp[i] + fp[i])
        if (tp[i] + fn[i] != 0) recall += t * tp[i].toDouble() / (tp[i] + fn[i])
    }
    precision /= sum
    recall /= sum
    return if (precision + recall == 0.0) {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fun main(args: Array<String>) {
    val test = "04"
//    for (test in listOf("01", "02", "03", "04", "05", "06", "07")) {
    val debug: Boolean
    val bufferedReader = if (args.isNotEmpty() && args.first() == "debug") {
        debug = true
        File("tests/$test").bufferedReader()
    } else {
        debug = false
        System.`in`.bufferedReader()
    }
    bufferedReader.use { it ->
        val features = it.readLine()?.toInt()!!
        val classes = it.readLine()?.toInt()!!
        val objects = it.readLine()?.toInt()!!
        val X = ArrayList<Vector>()
        val y = ArrayList<Int>()
        for (i in 1..objects) {
            val a = it.readLine()?.split(' ')?.map(String::toLong)!!
            X.add(a.dropLast(1) as Vector)
            y.add(a.last().toInt())
        }

        var bNeighbors = 10
        var bMetric = "euclidean"
        var bKernel = "triangular"
        var bestScore = -100.0
        for (neighbors in listOf(5, 4, 3, 2)) {
            for (kernel in listOf("triangular", "uniform", "parabolic", "biweight", "triweight", "tricube")) {
                for (metric in listOf("euclidean", "manhattan")) {
                    val clf = KNNClassifier(neighbors = neighbors, kernel = kernel, metric = metric)
                    val score = validate(clf, X, y, 5)
//                    print("$score ")
                    if (score > bestScore) {
                        bestScore = score
                        bNeighbors = neighbors
                        bMetric = metric
                        bKernel = kernel
                    }
                }
            }
        }
        println()

        val testSize = it.readLine()?.toInt()!!
        val testX = ArrayList<Vector>()
        for (i in 1..testSize) {
            testX.add(it.readLine()?.split(' ')?.map(String::toLong)!! as Vector)
        }

        val clf = KNNClassifier(neighbors = bNeighbors, metric = bMetric, kernel = bKernel)
        clf.fit(X, y)

        val predictions = clf.predict(testX)
        if (debug) {
            val lines = File("tests/$test.a").bufferedReader().readLines()
            val correct = lines.dropLast(1).map(String::toInt).toList()
            val fscore = lines.last().toDouble()
            val my = predictions.map { y[it.first().first] }
//            println((my zip correct).joinToString("\n"))

            println("""my score = ${fScore(correct, my)} threshold =  $fscore metric = $bMetric kernel = $bKernel neighbors = $bNeighbors""")
        } else {
            for (prediction in predictions) {
                print(prediction.size)
                println(prediction.joinToString(separator = " ", prefix = " ") { "%d %.3f".format(it.first + 1, it.second) })
            }
        }
    }
}
//}
