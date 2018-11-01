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
                else -> throw Exception("Bad kernel = $kernel")
            }
        }
        return 0.0
    }
}

fun crossValidation(clf: KNNClassifier, X: List<Vector>, y: List<Int>, cv: Int = 10): Double {
    val (partsX, partsY) = (X zip y).shuffled()
            .withIndex()
            .groupBy { it.index % cv }.values
            .map { it.map { indexed -> indexed.value } }
            .map { it.unzip() }
            .unzip()

    clf.fit(partsX.drop(1).flatten(), partsY.drop(1).flatten())
    return fScore(clf.predict(partsX.first()).map { partsY.drop(1).flatten()[it.first().first] }, partsY.first())
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
        precision += count[i] * (if (count[i] == 0) 0.0 else tp[i].toDouble() / count[i])
        recall += count[i] * (if (tp[i] + fn[i] == 0) 0.0 else tp[i].toDouble() / tp[i] + fn[i])
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
    System.`in`.bufferedReader().use { it ->
        //    File("tests/01").bufferedReader().use { it ->
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
        for (neighbors in listOf(1, 2, 3, 5, 7, 8, 10, 15, 20)) {
            for (kernel in listOf("uniform", "triangular", "parabolic", "biweight", "triweight")) {
                for (metric in listOf("euclidean", "manhattan")) {
                    val clf = KNNClassifier(neighbors = neighbors, kernel = kernel, metric = metric)
                    val score = crossValidation(clf, X, y)
                    if (score > bestScore) {
                        bestScore = score
                        bNeighbors = neighbors
                        bMetric = metric
                        bKernel = kernel
                    }
                }
            }
        }

        val testSize = it.readLine()?.toInt()!!
        val testX = ArrayList<Vector>()
        for (i in 1..testSize) {
            testX.add(it.readLine()?.split(' ')?.map(String::toLong)!! as Vector)
        }

        val clf = KNNClassifier(neighbors = bNeighbors, metric = bMetric, kernel = bKernel)
        clf.fit(X, y)

        val predictions = clf.predict(testX)
        for (prediction in predictions) {
            print(prediction.size)
            println(prediction.joinToString(separator = " ", prefix = " ") { "%d %.3f".format(it.first + 1, it.second) })
        }
    }
}
