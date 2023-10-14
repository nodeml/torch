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

  type<T extends TensorTypes>(type: T): Tensor<T>;

  dtype: TensorType;

  static fromTypedArray(data: ArrayTypes, shape: number[]);

  toMultiArray: () => MultiDimType<TensorType>;

  squeeze: (dim: number) => Tensor<TensorType>;

  unsqueeze: (dim: number) => Tensor<TensorType>;

  add: <T extends TensorTypes = typeof types.float>(
    a: Tensor<T> | number
  ) => Tensor;

  sub: <T extends TensorTypes = typeof types.float>(
    a: Tensor<T> | number
  ) => Tensor;

  mul: <T extends TensorTypes = typeof types.float>(
    a: Tensor<T> | number
  ) => Tensor;

  div: <T extends TensorTypes = typeof types.float>(
    a: Tensor<T> | number
  ) => Tensor;

  get: (...operators: TorchIndexOperators[]) => Tensor<TensorType>;

  set: <T extends TensorTypes = TensorType>(
    a: Tensor<T>,
    ...operators: TorchIndexOperators[]
  ) => void;

  clone: () => Tensor<TensorType>;

  matmul: <T extends TensorTypes = typeof types.float>(a: Tensor<T>) => Tensor;

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

export declare function greater<
  A extends TensorTypes = typeof types.float,
  B extends TensorTypes = typeof types.float
>(a: A, b: B | number): Tensor<typeof types.bool>;

export declare function greaterEqual<
  A extends TensorTypes = typeof types.float,
  B extends TensorTypes = typeof types.float
>(a: A, b: B | number): Tensor<typeof types.bool>;

export declare function less<
  A extends TensorTypes = typeof types.float,
  B extends TensorTypes = typeof types.float
>(a: A, b: B | number): Tensor<typeof types.bool>;

export declare function lessEqual<
  A extends TensorTypes = typeof types.float,
  B extends TensorTypes = typeof types.float
>(a: A, b: B | number): Tensor<typeof types.bool>;

export declare function equal<
  A extends TensorTypes = typeof types.float,
  B extends TensorTypes = typeof types.float
>(a: A, b: B | number): Tensor<typeof types.bool>;

export declare function zeros<T extends TensorTypes = typeof types.float>(
  shape: number[],
  dtype?: T
): Tensor<T>;

export declare function cat<T extends TensorTypes = typeof types.float>(
  tensors: Tensor<T>[]
): Tensor<T>;

export declare function where(condition: Tensor<typeof types.bool>): Tensor[];
