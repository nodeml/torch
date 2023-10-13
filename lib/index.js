"use strict";
const torch = require("bindings")("nodeml_torch");

torch.Tensor.prototype.toMultiArray = function () {
  const arr = Array.from(this.toArray());
  const shape = this.shape;
  const result = [];
  const totalElements = shape.reduce((acc, cur) => acc * cur, 1);

  for (let i = 0; i < totalElements; i++) {
    let position = result;
    let index = i;

    for (let j = shape.length - 1; j >= 0; j--) {
      const currentShape = shape[j];
      const currentIndex = index % currentShape;
      index = Math.floor(index / currentShape);

      if (position[currentIndex] === undefined) {
        position[currentIndex] = j === 0 ? arr[i] : [];
      }

      position = position[currentIndex];
    }
  }

  return result;
};

function tensor(data, shape) {
  return torch.Tensor.fromTypedArray(data, shape || [data.length]);
}

module.exports = { ...torch, tensor };
