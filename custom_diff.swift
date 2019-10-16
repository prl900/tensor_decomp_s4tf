import Glibc

func sillyExp(_ x: Float) -> Float {
    //let 𝑒 = Float(M_E)
    print("Taking 𝑒 to the power of \(x)!")
    //return pow(𝑒, x)
    return exp(x)
}

@differentiating(sillyExp)
func sillyDerivative(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
    let y = sillyExp(x)
    return (value: y, pullback: { v in v * y })
}

print("exp(3) =", sillyExp(3))
print("𝛁exp(3) =", gradient(of: sillyExp)(3))


let x: Float = 5.0
let y: Float = 3.0

print(gradient(at: x) { x in
    x*x
})

print(gradient(at: x, y) { x, y in
    x*x + withoutDerivative(at: y*y)
})

print(gradient(at: x, y) { x, y in
    x*x + y*y
})

print(gradient(of: { x, y in
    x*x + y*y
})(x, y))

print(gradient(at: x) { x in
    x*x + withoutDerivative(at: x*x)
})

print(gradient(at: x, y) { x, y in
    x*x + withoutDerivative(at: y*y)
})

print(gradient(at: x, y) { x, y in
    sin(sin(sin(x))) + withoutDerivative(at: cos(cos(cos(y))))
})

