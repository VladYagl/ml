import java.io.File
import java.util.*

fun main(args: Array<String>) {
    val random = Random()

    val a: Vector = arrayListOf(2.3, 1.337, 0.2, 1488.0)

    val n = 2000
    val f = 3
    val maxInt = 100_000
    val noise = 0.25

    File("res/input.txt").bufferedWriter().use {
        it.appendln("" + f)
        it.appendln("" + n)
        for (i in 1..n) {
            val x = (1..f).map { random.nextDouble() * maxInt * i } as Vector
            x.add(1.0)
            val y = a * x * (1 - random.nextDouble() * noise + noise / 2) + i
            it.appendln("" + x.str() + ' ' + y.toInt())
        }
    }

    solve(arrayOf("debug"))
}

fun Vector.str(): String {
    return this.dropLast(1).joinToString(separator = " ") { it.toInt().toString() }
}
