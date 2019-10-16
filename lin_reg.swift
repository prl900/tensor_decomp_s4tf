import TensorFlow

// Custom differentiable type.
struct LinearRegressor: Differentiable {
    var w: Tensor<Float>
    var b: Tensor<Float>
    func callAsFunction(_ x: Tensor<Float>) -> Tensor<Float> {
        return matmul(x, w) + b
    }
}

// Input: x shape [10, 1] (10 samples, 1 feature)
//let x: Tensor<Float> = [[0], [1], [2], [3], [4], [5], [6]]

// Output: same shape as x filled with small "random" deviations from 2*x+1
//let y: Tensor<Float> = [[0.90], [3.21], [4.77], [7.13], [9.02], [10.84], [13.23]]


/**** Curiously, adding an extra row to the data makes SGD non-convergent! ****/
let x: Tensor<Float> = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
let y: Tensor<Float> = [[0.90], [3.21], [4.77], [7.13], [9.02], [10.84], [13.23], [14.92], [16.93]]

// Weights and bias tensor initialization:
let w: Tensor<Float> = [[0.0]]
let b: Tensor<Float> = [[0.0]]

// Instantiate LinearRegressor Model with init values
var regressor = LinearRegressor(w: w, b: b)

// Declare SGD optimizer for model
let optimizer = SGD(for: regressor, learningRate: 0.01)

Context.local.learningPhase = .training

// SGD Training loop
for _ in 0..<1000 {
    let 𝛁model = regressor.gradient { regressor -> Tensor<Float> in
        let ŷ = regressor(x)
        //let loss = l2Loss(predicted: ŷ, expected: y)
        let loss = (ŷ - y).squared().mean()
        print("Loss: \(loss)")
        return loss
    }
    optimizer.update(&regressor, along: 𝛁model)
}

// Learned weights and bias for data
print("Learned weight tensor:, \(regressor.w)")
print("Learned bias tensor:, \(regressor.b)")
