import java.io.File
import kotlin.math.pow

typealias Vector = ArrayList<Int>

private var debug: Boolean = false

private var nodeCount = 0

private val nodes = ArrayList<DecisionTree.Node>()

class DecisionTree(val e: Double = 1e-8) {

    sealed class Node {
        val id = ++nodeCount

        init {
            nodes.add(this)
        }

        data class Leaf(val value: Int) : Node()

        data class Split(val feature: Int, val threshold: Double, val left: Node, val right: Node) : Node()

        final override fun toString(): String {
            return when (this) {
                is Leaf -> "C $value"
                is Split -> "Q ${feature + 1} $threshold ${nodeCount - left.id + 1} ${nodeCount - right.id + 1}"
            }
        }
    }

    private lateinit var root: Node
    private var features: Int = 0
    private var classes: Int = 0

    fun fit(X: List<Vector>, y: List<Int>, features: Int, classes: Int) {
        this.features = features
        this.classes = classes
        root = buildTree(X zip y)
    }

    private fun buildTree(list: List<Pair<Vector, Int>>, death: Int = 1): DecisionTree.Node {
        val (_, y) = list.unzip()

        val hist = HashMap<Int, Int>()
        for (c in y) {
            hist[c] = hist[c]?.plus(1) ?: 1
        }
        val before = gini(hist, y.size)
        if (before <= e || list.size == 1 || death == 11) {
            return Node.Leaf(hist.maxBy { it.value }!!.key)
        }

        var bestGain = -1.0
        var bestFeature = -1
        var bestThreshold = 0.0

        for (feature in 0 until features) {
            var lsize = 0
            var rsize = y.size
            val left = HashMap<Int, Int>()
            val right = HashMap<Int, Int>()
            right.putAll(hist)

            for ((x, c) in list.sortedBy { it.first[feature] }.dropLast(1)) {
                lsize += 1
                rsize -= 1
                left[c] = left[c]?.plus(1) ?: 1
                right[c] = right[c]!! - 1
                val gain = before - gini(left, lsize) * lsize / y.size - gini(right, rsize) * rsize / y.size
                if (gain > bestGain) {
                    bestGain = gain
                    bestFeature = feature
                    bestThreshold = x[feature] + 0.5
                }
            }
        }

        return Node.Split(
            bestFeature, bestThreshold,
            buildTree(list.filter { it.first[bestFeature] < bestThreshold }, death + 1),
            buildTree(list.filter { it.first[bestFeature] >= bestThreshold }, death + 1)
        )
    }

    private fun gini(hist: HashMap<Int, Int>, size: Int): Double {
        return 1 - hist.values.sumByDouble { (it.toDouble() / size).pow(2) }
    }

    override fun toString(): String {
        return nodes.reversed().joinToString(separator = "\n", prefix = "$nodeCount\n")
    }
}

fun main(args: Array<String>) {
    val bufferedReader = if (args.isNotEmpty() && args.first() == "debug") {
        debug = true
        File("res/input.txt").bufferedReader()
    } else {
        debug = false
        System.`in`.bufferedReader()
    }

    bufferedReader.use { it ->
        val (features, classes) = it.readLine().split(' ').map(String::toInt)
        val n = it.readLine().toInt()
        val X = ArrayList<Vector>()
        val y = ArrayList<Int>()
        for (i in 1..n) {
            val temp = Vector()
            val a = it.readLine().split(' ').map(String::toInt)
            temp.addAll(a.dropLast(1))
            X.add(temp)
            y.add(a.last())
        }

        val dt = DecisionTree()
        dt.fit(X, y, features, classes)
        println(dt.toString())
    }
}