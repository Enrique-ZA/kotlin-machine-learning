import kotlin.random.Random

class Gate {
    var x: Matrix = matrixCreate(0, 0) 

    var w1: Matrix = matrixCreate(0, 0)
    var b1: Matrix = matrixCreate(0, 0)
    var a1: Matrix = matrixCreate(0, 0)

    var w2: Matrix = matrixCreate(0, 0)
    var b2: Matrix = matrixCreate(0, 0)
    var a2: Matrix = matrixCreate(0, 0)

    var expected: Array<Double>? = null
}

fun gateCreate(x: Matrix, w1: Matrix, b1: Matrix, a1: Matrix, w2: Matrix, b2: Matrix, a2: Matrix): Gate {
    return Gate().apply{
        this.x  = x 
        this.w1 = w1 
        this.b1 = b1 
        this.a1 = a1 
        this.w2 = w2 
        this.b2 = b2 
        this.a2 = a2
    }
}

fun gateLoss(gate: Gate, ti: Matrix, to: Matrix): Double {
    if (ti.rows != to.rows || to.cols != gate.a2.cols) {
        throw Error("loss error: ti.rows != to.rows || to.cols != gate.a2.cols")
    }
    var result: Double = 0.0
    for (i in 0 until ti.rows) {
        val rowMatrix1 = matrixRow(ti, i)
        val rowMatrix2 = matrixRow(to, i)

        gate.x = matrixCopy(gate.x, rowMatrix1)
        gateForward(gate)

        for (j in 0 until to.cols) {
            val dist = gate.a2.samples[gate.a2.cols * 0 + j] - rowMatrix2.samples[rowMatrix2.cols * 0 + j]
            result += (dist * dist)
        }
    }
    return result / ti.rows
}

fun gateLearn(gate: Gate, gradient: Gate, rate: Double): Unit {
    for(i in 0 until gate.w1.rows){
        for(j in 0 until gate.w1.cols){
            gate.w1.samples[gate.w1.cols * i + j] -= gradient.w1.samples[gradient.w1.cols * i + j] * rate
        }
    }

    for(i in 0 until gate.b1.rows){
        for(j in 0 until gate.b1.cols){
            gate.b1.samples[gate.b1.cols * i + j] -= gradient.b1.samples[gradient.b1.cols * i + j] * rate
        }
    }

    for(i in 0 until gate.w2.rows){
        for(j in 0 until gate.w2.cols){
            gate.w2.samples[gate.w2.cols * i + j] -= gradient.w2.samples[gradient.w2.cols * i + j] * rate
        }
    }

    for(i in 0 until gate.b2.rows){
        for(j in 0 until gate.b2.cols){
            gate.b2.samples[gate.b2.cols * i + j] -= gradient.b2.samples[gradient.b2.cols * i + j] * rate
        }
    }
}

fun gateFiniteDiff(gate: Gate, gradient: Gate, epsilon: Double, ti: Matrix, to: Matrix): Unit {
    var saved: Double
    val loss = gateLoss(gate, ti, to)

    for(i in 0 until gate.w1.rows){
        for(j in 0 until gate.w1.cols){
            saved = gate.w1.samples[gate.w1.cols * i + j]
            gate.w1.samples[gate.w1.cols * i + j] += epsilon
            gradient.w1.samples[gradient.w1.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon
            gate.w1.samples[gate.w1.cols * i + j] = saved
        }
    }

    for(i in 0 until gate.b1.rows){
        for(j in 0 until gate.b1.cols){
            saved = gate.b1.samples[gate.b1.cols * i + j]
            gate.b1.samples[gate.b1.cols * i + j] += epsilon
            gradient.b1.samples[gradient.b1.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon
            gate.b1.samples[gate.b1.cols * i + j] = saved
        }
    }

    for(i in 0 until gate.w2.rows){
        for(j in 0 until gate.w2.cols){
            saved = gate.w2.samples[gate.w2.cols * i + j]
            gate.w2.samples[gate.w2.cols * i + j] += epsilon
            gradient.w2.samples[gradient.w2.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon
            gate.w2.samples[gate.w2.cols * i + j] = saved
        }
    }

    for(i in 0 until gate.b2.rows){
        for(j in 0 until gate.b2.cols){
            saved = gate.b2.samples[gate.b2.cols * i + j]
            gate.b2.samples[gate.b2.cols * i + j] += epsilon
            gradient.b2.samples[gradient.b2.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon
            gate.b2.samples[gate.b2.cols * i + j] = saved
        }
    }
}

fun gateForward(gate: Gate): Gate {
    gate.a1 = matrixMult     (gate.a1, gate.x,     gate.w1)
    gate.a1 = matrixSum      (gate.a1, gate.b1)
    gate.a1 = matrixSigmoidf (gate.a1)

    gate.a2 = matrixMult     (gate.a2, gate.a1,    gate.w2)
    gate.a2 = matrixSum      (gate.a2, gate.b2)
    gate.a2 = matrixSigmoidf (gate.a2)
    return gate
}

data class Matrix(
    var rows: Int,
    var cols: Int,
    var stride: Int,
    var samples: Array<Double>
)

fun matrixCreate(numRows: Int, numCols: Int): Matrix {
    val size = numRows * numCols
    val arr: Array<Double> = Array<Double>(size) { 0.0 }
    return Matrix(
        numRows,
        numCols,
        numCols,
        arr
    )
}

fun matrixRandomize(mat: Matrix, low: Double, high: Double): Matrix {
    for (i in 0 until mat.rows) {
        for (j in 0 until mat.cols) {
            mat.samples[mat.cols * i + j] = Random.nextDouble(low, high)
        }
    }
    return mat
}

fun matrixSum(org: Matrix, other: Matrix): Matrix {
    if (other.rows != org.rows || other.cols != org.cols) {
        throw Error("matrix sum error: other and org must have the same size")
    }
    for(i in 0 until other.rows){
        for(j in 0 until other.cols){
            org.samples[other.cols * i + j] += other.samples[other.cols * i + j]
        }
    }
    return org
}

fun matrixFill(mat: Matrix, num: Double): Matrix {
    for(i in 0 until mat.rows){
        for(j in 0 until mat.cols){
            mat.samples[mat.cols * i + j] = num
        }
    }
    return mat
}

fun matrixMult(dst: Matrix, a: Matrix, b: Matrix): Matrix {
    if(a.cols != b.rows){
        throw Error("mult error 1: for param2 and param3, the rows do not match")
    }

    if(dst.rows != a.rows || dst.cols != b.cols){
        throw Error("mult error 2: for params, either the rows of param1 and param2 or cols of param1 and param3 do not match")
    }

    val n = a.cols
    for(i in 0 until dst.rows){
        for(j in 0 until dst.cols){
            dst.samples[dst.cols * i + j] = 0.0
            for(k in 0 until n){
                dst.samples[dst.cols * i + j] += a.samples[a.cols * i + k] * b.samples[b.cols * k + j]
            }
        }
    }
    return dst
}

fun sigmoidf(x: Double): Double {
    return 1.0 / (1.0 + Math.exp(-x))
}

fun matrixSigmoidf(mat: Matrix): Matrix {
    for (i in 0 until mat.rows) {
        for (j in 0 until mat.cols) {
            mat.samples[mat.cols * i + j] = sigmoidf(mat.samples[mat.cols * i + j])
        }
    }
    return mat
}

fun matrixRow(mat: Matrix, row: Int): Matrix {
    val startIndex = row * mat.cols
    val endIndex = startIndex + mat.cols
    val rowSamples = mat.samples.sliceArray(startIndex until endIndex)
    return Matrix(
        1,
        mat.cols,
        mat.cols,
        rowSamples
    )
}

fun matrixCopy(dst: Matrix, src: Matrix): Matrix {
    if(dst.rows != src.rows || dst.cols != src.cols){
        throw Error("copy error: matrices don't match")
    }
    dst.samples = src.samples.copyOf()
    return dst
}

fun matrixSlice(arr: Array<Double>, rows: Int, cols: Int, step: Int, start: Int): Array<Double> {
    val temp: Array<Double> = Array<Double>(rows * cols) { 0.0 }
    var index = start
    for(i in 0 until rows){
        for(j in 0 until cols){
            if (index < arr.size) {
                temp[cols * i + j] = arr[index]
                index++
            }
        }
        index += step - cols
    }
    return temp
}

fun main(){
    var gate = gateCreate(
        matrixCreate(1, 2),
        matrixCreate(2, 2),
        matrixCreate(1, 2),
        matrixCreate(1, 2),
        matrixCreate(2, 1),
        matrixCreate(1, 1),
        matrixCreate(1, 1)
    )

    var gradient = gateCreate(
        matrixCreate(1, 2),
        matrixCreate(2, 2),
        matrixCreate(1, 2),
        matrixCreate(1, 2),
        matrixCreate(2, 1),
        matrixCreate(1, 1),
        matrixCreate(1, 1)
    )

    // xor
    gate.expected = arrayOf(
        0.0, 0.0, 0.0, 
        0.0, 1.0, 1.0, 
        1.0, 0.0, 1.0, 
        1.0, 1.0, 0.0
    )

    val stride = 3
    val expectedLength = gate.expected!!.size

    val tiArr: Array<Double> = matrixSlice(gate.expected!!, expectedLength/stride, 2, stride, 0)
    val toArr: Array<Double> = matrixSlice(gate.expected!!, expectedLength/stride, 1, stride, 2)

    val ti: Matrix = matrixCreate(expectedLength/stride, 2).apply {
        this.samples = tiArr
        this.stride = stride
    }

    val to: Matrix = matrixCreate(expectedLength/stride, 1).apply {
        this.samples = toArr
        this.stride = stride
    }

    matrixRandomize(gate.w1, 0.0, 1.0)
    matrixRandomize(gate.b1, 0.0, 1.0)
    matrixRandomize(gate.w2, 0.0, 1.0)
    matrixRandomize(gate.b2, 0.0, 1.0)

    val epsilon = 1e-1
    val rate = 1e-1

    for(i in 0 until 50*1000){
        gateFiniteDiff(gate, gradient, epsilon, ti, to)
        gateLearn(gate, gradient, rate)
    }

    for(i in 0 until 2){
        for(j in 0 until 2){
            gate.x.samples = arrayOf(i.toDouble(), j.toDouble())
            gate = gateForward(gate)
            val y = gate.a2.samples[0]
            println("$i ^ $j = $y")
        }
    }

}
