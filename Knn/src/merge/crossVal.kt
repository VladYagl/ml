package merge

import java.lang.Integer.max
import java.lang.Exception

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