"use strict";
const torch = require("bindings")("nodeml_torch");

function makeTypedArray(dtype,data) {
  switch (dtype) {
    case types.int32:
      return new Int32Array(data);
    case types.double:
      return new Float64Array(data);
    case types.float:
      return new Float32Array(data);

    case types.uint8:
      return new Uint8Array(data);
    case types.long:
      return new BigInt64Array(data);
    default:
      throw new Error("No typed array for type " + dtype)
  }
}

torch.Tensor.prototype[Symbol.iterator] = function * () {
  const shape = this.shape;
  for(let i = 0; i < shape[0]; i++){
    yield this.get(i)
  }
}

// torch.Tensor.prototype.toMultiArray = function () {
//   const arr = Array.from(this.toArray());
//   const shape = this.shape;
//   const result = [];
//   const totalElements = shape.reduce((acc, cur) => acc * cur, 1);

//   for (let i = 0; i < totalElements; i++) {
//     let position = result;
//     let index = i;

//     for (let j = shape.length - 1; j >= 0; j--) {
//       const currentShape = shape[j];
//       const currentIndex = index % currentShape;
//       index = Math.floor(index / currentShape);

//       if (position[currentIndex] === undefined) {
//         position[currentIndex] = j === 0 ? arr[i] : [];
//       }

//       position = position[currentIndex];
//     }
//   }

//   return result;
// };

function flattenArray(arr) {
  const shape = [arr.length]
  while (arr.length > 0 && Array.isArray(arr[0])) {
    shape.push(arr[0].length)
    // Forcing Array<any> is required to make TS happy; otherwise, TS2349
    arr = arr.reduce((acc, val) => acc.concat(val), [])

    // Running assert in every step to make sure that shape is regular
    const numel = shape.reduce((acc, cur) => acc * cur)
    assert.strictEqual(arr.length, numel)
  }
  return {
    data: arr,
    shape
  }
}


function tensor(data, shape) {
  if (ArrayBuffer.isView(data)) {
    return torch.Tensor.fromTypedArray(data, shape || [data.length]);
  }
  else {
    const payload = flattenArray(data)
    const tensorShape = shape || payload.shape
    if(typeof payload.data[0] == 'boolean'){
      return torch.Tensor.fromTypedArray(makeTypedArray('int32',payload.data.map(c => c ? 1 : 0)),tensorShape).type(types.bool)
    }
    else
    {
      return torch.Tensor.fromTypedArray(makeTypedArray('float',payload.data),tensorShape)
    }
  }
}

module.exports = { ...torch, tensor };
