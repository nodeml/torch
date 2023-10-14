export type ArrayTypes = Float32Array | Int32Array | Uint8Array | BigInt64Array;

export type MiltiDim = Array<number[] | MiltiDim>;

type NullableNumber = number | null;
// https://stackoverflow.com/a/60059664
export type TorchIndexOperators =
  | number
  | null
  | "..."
  | []
  | [NullableNumber, NullableNumber]
  | [NullableNumber, NullableNumber, NullableNumber]
  | boolean;

export const types = {
  int32: "int32",
  double: "double",
  float: "float",
  uint8: "uint8",
  long: "long",
} as const;

export type TensorTypes = typeof types[keyof typeof types];

export declare class Tensor {
  shape: number[];

  toArray: () => ArrayTypes;

  reshape: (view: number[]) => Tensor;

  slice: (dim: number, start?: number, stop?: number) => Tensor;

  type(type: TensorTypes): Tensor;

  dtype: TensorTypes;

  static fromTypedArray(data: ArrayTypes, shape: number[]);

  toMultiArray: () => MiltiDim;

  squeeze: (dim: number) => Tensor;

  unsqueeze: (dim: number) => Tensor;

  add: (a: Tensor | number) => Tensor;

  sub: (a: Tensor | number) => Tensor;

  mul: (a: Tensor | number) => Tensor;

  div: (a: Tensor | number) => Tensor;

  get: (...operators: TorchIndexOperators) => Tensor;

  set: (a: Tensor, ...operators: TorchIndexOperators) => void;

  clone: () => Tensor;

  matmul: (a: Tensor) => Tensor;
}

export declare function rand(shape: number[], dtype?: TensorTypes): Tensor;

export declare function arange(end: number, dtype?: TensorTypes): Tensor;

export declare function arange(
  start: number,
  end: number,
  dtype?: TensorTypes
): Tensor;

export declare function arange(
  start: number,
  end: number,
  step: number,
  dtype?: TensorTypes
): Tensor;

export declare function tensor(data: ArrayTypes, shape?: number[]): Tensor;

export namespace nn {
  namespace functional {
    declare function interpolate(tensor: Tensor, size: number[]): Tensor;
  }
}

export namespace jit {
  declare class Module<OutputType = Tensor> {
    forward: (...args: Tensor) => Promise<OutputType>;
  }

  declare function load<OutputType = Tensor>(path: string): Module<OutputType>;
}
