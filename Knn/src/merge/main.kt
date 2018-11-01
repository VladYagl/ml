package merge

import java.io.File

fun main(args: Array<String>) {
    File("tests/01").bufferedReader().use { it ->
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
            println(prediction.joinToString(separator = " ", prefix = " ") { "" + it.first + " " + it.second })
        }
    }
}
