export type ArrayTypes = Float32Array | Int32Array | Uint8Array | BigInt64Array;

export type MiltiDim = Array<number[] | MiltiDim>;

export const types = {
  int32: "int32",
  double: "double",
  float: "float",
  uint8: "uint8",
  long: "long",
} as const;

export type TensorTypes = typeof types[keyof typeof types];

export declare class Tensor {
  shape: () => number[];

  toArray: () => ArrayTypes;

  reshape: (view: number[]) => Tensor;

  slice: (dim: number, start?: number, stop?: number) => Tensor;

  type(type: TensorTypes): Tensor;

  dtype: () => TensorTypes;

  static fromTypedArray(data: ArrayTypes, shape: number[]);

  toMultiArray: () => MiltiDim;
}

export declare function rand(shape: number[], dtype?: TensorTypes): Tensor;

export declare function tensor(
  data: ArrayTypes,
  shape?: number[]
): torch.Tensor;
