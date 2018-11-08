@file:Suppress("LocalVariableName")

import java.io.File
import java.lang.Double.max
import java.lang.Exception
import java.lang.Integer.min
import java.util.*
import kotlin.math.pow

typealias Vector = ArrayList<Double>

var debug: Boolean = false

operator fun Vector.times(other: Vector): Double {
    return (this zip other).sumByDouble { (a, b) -> a * b }
}

operator fun Vector.times(k: Double): Vector {
    return this.map { it * k } as Vector
}

operator fun Double.times(a: Vector): Vector {
    return a * this
}

operator fun Vector.plus(other: Vector): Vector {
    return (this zip other).map { (a, b) -> a + b } as Vector
}

operator fun Vector.minus(other: Vector): Vector {
    return (this zip other).map { (a, b) -> a - b } as Vector
}

fun main(args: Array<String>) {
    solve(args)
}

fun gradientDescend(
    W: Vector,
    X: List<Vector>,
    Y: Vector,
    iterations: Int,
    step: Double = 0.01,
    k: Int = 20,
    type: Int = 3
): Pair<Vector, Double> {
    val startTime = System.currentTimeMillis()
    val random = Random()

    val n = X.size
    val features = X[0].size
    var w = W
    val zip = X zip Y
    val parts = (n + k - 1) / k
    for (shit in 0..(iterations / features)) {
        var grad: Vector
        when (type) {
            1 -> {
                val x = X[shit % n]
                val y = Y[shit % n]
                grad = (w * x - y) * x
            }


            2 -> {
                val sub = zip.subList(shit % parts * k, min((shit % parts + 1) * k, zip.size))
                grad = sub.map { (x, y) ->
                    (w * x - y) * x
                }.reduce { acc, vector -> acc + vector }
                grad *= (1.0 / sub.size)
            }

            3 -> {
                grad = Vector(features)
                for (i in 0..features) {
                    grad.add(0.0)
                }
                for (s in (1..k)) {
                    val i = random.nextInt(n)
                    grad = grad + (w * X[i] - Y[i]) * X[i]
                }
            }

            else -> throw Exception("WTF??")
        }

        grad *= (1.0 / k)

        w = w - step * grad
    }

    val diff = (X zip Y).map { (x, y) ->
        (y - w * x).pow(2)
    }.reduce { acc, vector -> acc + vector }
    if (debug) {
        println("cost = " + diff * 1.0 / n)

        val stopTime = System.currentTimeMillis()
        val elapsedTime = stopTime - startTime
        println("time: " + elapsedTime / 1000.0)
        println()
    }


    if (w[1].isNaN() || w[1].isInfinite()) throw Exception()

    assert(w.size == features + 1)

    return Pair(w, diff)
}

fun solve(args: Array<String>) {
    val bufferedReader = if (args.isNotEmpty() && args.first() == "debug") {
        debug = true
        File("res/input.txt").bufferedReader()
    } else {
        debug = false
        System.`in`.bufferedReader()
    }
    bufferedReader.use { it ->
        val features = it.readLine().toInt()
        val n = it.readLine().toInt()
        var X = ArrayList<Vector>()
        var Y = ArrayList<Double>()
        var scale = 1.0
        for (i in 1..n) {
            val temp = Vector()
            val a = it.readLine().split(' ').map(String::toDouble)
            temp.addAll(a.dropLast(1))
            scale = max(scale, a.max()!!)
            X.add(temp)
            Y.add(a.last())
        }

        val oldX = X
        oldX.forEach { it.add(1.0) }
        val oldY = Y

        X = X.map { it * (1.0 / scale) } as ArrayList<Vector>
        Y = Y.map { it * (1.0 / scale) } as ArrayList<Double>
        X.forEach { it.add(1.0) }

        val w = Vector(features)
        val random = Random()
        for (i in 0..features) {
            w.add(random.nextDouble() / n - 1 / (2 * n))
        }

        val small = 100
        val (smallX, smallY) = (X zip Y).shuffled().take(small).unzip()
        val (first, _) = gradientDescend(w, smallX, smallY as Vector, iterations = 1000)

        val (ans1, cost1) = gradientDescend(first, X, Y, iterations = 200_000, step = 0.01)
        val (ans2, cost2) = gradientDescend(first, X, Y, iterations = 200_000, step = 0.001)

        val ans: Vector
        ans = if (cost1 < cost2) {
            ans1
        } else {
            ans2
        }

        ans[ans.size - 1] = ans.last() * scale

        println(ans.joinToString(separator = "\n") { "%.4f".format(it) })
    }
}

