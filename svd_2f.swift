import TensorFlow
import Foundation

// Custom differentiable type.
struct Decomposer: Differentiable {
    var p: Tensor<Float>
    var q: Tensor<Float>
    func callAsFunction() -> Tensor<Float> {
        return matmul(p, q)
    }
}

//let A: Tensor<Float> = [[0,1,2],[3,_TensorElementLiteral(floatLiteral:Float.nan),5],[6,7,8]]
//let A: Tensor<Float> = [[0,1,2],[3,nan,5],[6,7,8]]
let A: Tensor<Float> = [[0,1,2],[3,40,5],[6,7,8]]

let p: Tensor<Float> = [[1,1],[1,1],[1,1]]
let q: Tensor<Float> = [[1,1,1],[1,1,1]]

var model = Decomposer(p: p, q: q)
print(model())
print(A)

let optimizer = SGD(for: model, learningRate: 0.01)

// SGD Training loop
for _ in 0..<1000 {
    // p updating step
    var 𝛁model = model.gradient { model -> Tensor<Float> in
	let Â = model()
	let loss = (Â-A).squared().mean()
        return loss
    }
    𝛁model.q *= [[0,0,0],[1,1,1]]
    optimizer.update(&model, along: 𝛁model)
    
    𝛁model = model.gradient { model -> Tensor<Float> in
	let Â = model()
	let loss = (Â-A).squared().mean()
        return loss
    }
    𝛁model.q *= [[1,1,1],[0,0,0]]
    optimizer.update(&model, along: 𝛁model)
    
    // q updating step
    𝛁model = model.gradient { model -> Tensor<Float> in
	let Â = model()
	let loss = (Â-A).squared().mean()
        print("Loss: \(loss)")
        return loss
    }
    𝛁model.p *= [[0,1],[0,1],[0,1]]
    optimizer.update(&model, along: 𝛁model)
    	
    𝛁model = model.gradient { model -> Tensor<Float> in
	let Â = model()
	let loss = (Â-A).squared().mean()
        print("Loss: \(loss)")
        return loss
    }
    𝛁model.p *= [[1,0],[1,0],[1,0]]
    optimizer.update(&model, along: 𝛁model)
}

print("Learned p tensor:, \(model.p)")
print("Learned q tensor:, \(model.q)")
print(model())
