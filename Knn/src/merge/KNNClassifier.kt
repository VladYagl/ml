package merge

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