export type ArrayTypes =
  | Float32Array
  | Int32Array
  | Uint8Array
  | BigInt64Array
  | boolean[];

export type MultiDim<T = number> = Array<T[] | MultiDim<T>>;

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
  bool: "bool",
} as const;

export type TensorTypes = typeof types[keyof typeof types];

export type ArrayTypeToTensorType<T extends ArrayTypes> = T extends Float32Array
  ? typeof types.float
  : T extends Float64Array
  ? typeof types.double
  : T extends Int32Array
  ? typeof types.int32
  : T extends Uint8Array
  ? typeof types.uint8
  : T extends BigInt64Array
  ? typeof types.long
  : T extends boolean[]
  ? typeof types.bool
  : TensorTypes;

export type TensorTypeToArrayType<T extends TensorTypes> =
  T extends typeof types.float
    ? Float32Array
    : T extends typeof types.int32
    ? Int32Array
    : T extends typeof types.double
    ? Float64Array
    : T extends typeof types.uint8
    ? Uint8Array
    : T extends typeof types.long
    ? BigInt64Array
    : T extends typeof types.bool
    ? boolean[]
    : ArrayTypes;

export type MultiDimType<T extends TensorTypes> = T extends typeof types.bool
  ? MultiDim<boolean>
  : MultiDim<number>;

export declare class Tensor<TensorType extends TensorTypes = TensorTypes> {
  shape: number[];

  toArray: () => TensorTypeToArrayType<TensorType>;

  reshape: (view: number[]) => Tensor<TensorType>;

  slice: (dim: number, start?: number, stop?: number) => Tensor<TensorType>;

  type<T extends TensorTypes>(type: T): Tensor<T>;

  dtype: TensorType;

  static fromTypedArray(data: ArrayTypes, shape: number[]);

  toMultiArray: () => MultiDimType<TensorType>;

  squeeze: (dim: number) => Tensor<TensorType>;

  unsqueeze: (dim: number) => Tensor<TensorType>;

  add: (a: Tensor | number) => Tensor;

  sub: (a: Tensor | number) => Tensor;

  mul: (a: Tensor | number) => Tensor;

  div: (a: Tensor | number) => Tensor;

  get: (...operators: TorchIndexOperators) => Tensor<TensorType>;

  set: (a: Tensor<TensorType>, ...operators: TorchIndexOperators) => void;

  clone: () => Tensor<TensorType>;

  matmul: (a: Tensor) => Tensor;

  amax: (dim: number) => Tensor<TensorType>;
}

export declare function tensor<T extends ArrayTypes = ArrayTypes>(
  data: T,
  shape?: number[]
): Tensor<ArrayTypeToTensorType<T>>;

export namespace nn {
  namespace functional {
    declare function interpolate<T extends TensorTypes>(
      tensor: Tensor<T>,
      size: number[]
    ): Tensor<T>;
  }
}

export namespace jit {
  declare class Module<OutputType = Tensor> {
    forward: (...args: Tensor[]) => Promise<OutputType>;
  }

  declare function load<OutputType = Tensor>(path: string): Module<OutputType>;
}

export declare function rand<T extends TensorTypes = typeof types.float>(
  shape: number[],
  dtype?: T
): Tensor<T>;

export declare function arange<T extends TensorTypes = typeof types.float>(
  end: number,
  dtype?: T
): Tensor<T>;

export declare function arange<T extends TensorTypes = typeof types.float>(
  start: number,
  end: number,
  dtype?: T
): Tensor<T>;

export declare function arange<T extends TensorTypes = typeof types.float>(
  start: number,
  end: number,
  step: number,
  dtype?: T
): Tensor<T>;

export declare function greater(
  a: Tensor,
  b: Tensor
): Tensor<typeof types.bool>;

export declare function greaterEqual(
  a: Tensor,
  b: Tensor
): Tensor<typeof types.bool>;

export declare function less(a: Tensor, b: Tensor): Tensor<typeof types.bool>;

export declare function lessEqual(
  a: Tensor,
  b: Tensor
): Tensor<typeof types.bool>;

export declare function equal(a: Tensor, b: Tensor): Tensor<typeof types.bool>;

export declare function zeros<T extends TensorTypes = typeof types.float>(
  shape: number[],
  dtype?: T
): Tensor<T>;

export declare function cat(tensors: Tensor[]): Tensor;

export declare function where(condition: Tensor): Tensor[];
