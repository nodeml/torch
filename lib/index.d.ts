declare namespace torch {

  type ArrayTypes =
    | Float32Array
    | Int32Array
    | Uint8Array
    | BigInt64Array
    | boolean[] | number[];

  type MultiDim<T = number> = Array<T[] | MultiDim<T>>;

  type NullableNumber = number | null;
  // https://stackoverflow.com/a/60059664
  type TorchIndexOperators =
    | number
    | null
    | "..."
    | []
    | [NullableNumber, NullableNumber]
    | [NullableNumber, NullableNumber, NullableNumber]
    | boolean
    | Tensor;

  declare const types: {
    int32: "int32",
    double: "double",
    float: "float",
    uint8: "uint8",
    long: "long",
    bool: "bool",
  };

  type TensorTypes = typeof types[keyof typeof types];
  type InterpolationModes = 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
  type ArrayTypeToTensorType<T extends ArrayTypes> = T extends Float32Array
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
    : T extends number[] ? typeof types.float
    : TensorTypes;

  type TensorTypeToArrayType<T extends TensorTypes> =
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

  type MultiDimType<T extends TensorTypes> = T extends typeof types.bool
    ? MultiDim<boolean>
    : MultiDim<number>;

  declare class Tensor<TensorType extends TensorTypes = TensorTypes> {
    shape: number[];

    toArray: () => TensorTypeToArrayType<TensorType>;

    reshape: (view: number[]) => Tensor<TensorType>;

    transpose: (dim0: number, dim1: number) => Tensor<TensorType>;

    type<T extends TensorTypes>(type: T): Tensor<T>;

    dtype: TensorType;

    static fromTypedArray(data: ArrayTypes, shape: number[]);

    // toMultiArray: () => MultiDimType<TensorType>;

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

    set: <T extends TensorTypes = TensorType>(value: Tensor<T>,
      ...ops: [TorchIndexOperators, ...TorchIndexOperators[]]
    ) => void;

    clone: () => Tensor<TensorType>;

    matmul: <T extends TensorTypes = typeof types.float>(a: Tensor<T>) => Tensor;

    amax: (dim: number) => Tensor<TensorType>;

    split: (s: number | number[], dim?: number) => Tensor<TensorType>[];

    argsort: (dim: number, decending: boolean = false) => Tensor<TensorType>;

    max: (
      dim: number,
      keepDim: boolean = false
    ) => [Tensor<TensorType>, Tensor<typeof types.int32>];

    view: (...dims: number[]) => Tensor<TensorType>;

    any: (
      dim: number,
      keepDim: boolean = false
    ) => Tensor<
      TensorType extends typeof types.uint8 ? TensorType : typeof types.bool
    >;

    clamp: (min: number, max: number) => Tensor<TensorType>;

    sigmoid: () => Tensor<TensorType>;

    cuda: () => Tensor<TensorType>;

    cpu: () => Tensor<TensorType>;

    detach: () => Tensor<TensorType>;

    backward: () => void;

    *[Symbol.iterator](): IterableIterator<Tensor<TensorType>>;
  }

  declare function tensor<T extends ArrayTypes = ArrayTypes>(
    data: T,
    shape?: number[]
  ): Tensor<ArrayTypeToTensorType<T>>;

  declare function rand<T extends TensorTypes = typeof types.float>(
    shape: number[],
    dtype?: T
  ): Tensor<T>;

  declare function arange<T extends TensorTypes = typeof types.float>(
    end: number,
    dtype?: T
  ): Tensor<T>;

  declare function arange<T extends TensorTypes = typeof types.float>(
    start: number,
    end: number,
    dtype?: T
  ): Tensor<T>;

  declare function arange<T extends TensorTypes = typeof types.float>(
    start: number,
    end: number,
    step: number,
    dtype?: T
  ): Tensor<T>;

  declare function greater<
    A extends TensorTypes = typeof types.float,
    B extends TensorTypes = typeof types.float
  >(a: Tensor<A>, b: Tensor<B> | number): Tensor<typeof types.bool>;

  declare function greaterEqual<
    A extends TensorTypes = typeof types.float,
    B extends TensorTypes = typeof types.float
  >(a: Tensor<A>, b: Tensor<B> | number): Tensor<typeof types.bool>;

  declare function less<
    A extends TensorTypes = typeof types.float,
    B extends TensorTypes = typeof types.float
  >(a: Tensor<A>, b: Tensor<B> | number): Tensor<typeof types.bool>;

  declare function lessEqual<
    A extends TensorTypes = typeof types.float,
    B extends TensorTypes = typeof types.float
  >(a: Tensor<A>, b: Tensor<B> | number): Tensor<typeof types.bool>;

  declare function equal<
    A extends TensorTypes = typeof types.float,
    B extends TensorTypes = typeof types.float
  >(a: Tensor<A>, b: Tensor<B> | number): Tensor<typeof types.bool>;

  declare function zeros<T extends TensorTypes = typeof types.float>(
    shape: number[],
    dtype?: T
  ): Tensor<T>;

  declare function cat<T extends TensorTypes = typeof types.float>(
    tensors: Tensor<T>[],
    dim: number = 0
  ): Tensor<T>;

  declare function where(condition: Tensor<typeof types.bool>): Tensor[];

  declare function chunk<T extends TensorTypes = typeof types.float>(tensor: Tensor<T>, chunks: number, dim?: number): Tensor<T>[];

  declare function empty<T extends TensorTypes = typeof types.float>(
    shape: number[],
    dtype?: T
  ): Tensor<T>;

  declare function emptyLike<T extends TensorTypes>(
    tensor: Tensor<T>
  ): Tensor<T>;

  declare function runBlockingAsync<Result = unknown, Args extends unknown[]>(func: (...args: Args) => Result, ...args: Args): Promise<Result>

  namespace nn {
    namespace functional {
      declare function interpolate<T extends TensorTypes>(
        tensor: Tensor<T>,
        size: number[],
        mode: InterpolationModes,
        options?: {
          alignCorners?: boolean;
          antiAlias?: boolean;
        }
      ): Tensor<T>;

      declare function pad<T extends TensorTypes>(
        tensor: Tensor<T>,
        pad: [number, number]
      ): Tensor<T>;
      declare function pad<T extends TensorTypes>(
        tensor: Tensor<T>,
        pad: [number, number, number, number]
      ): Tensor<T>;
    }
  }

  namespace jit {
    declare class Module<OutputType = Tensor> {
      forward: (...args: Tensor[]) => Promise<OutputType>;
    }

    declare function load<OutputType = Tensor>(
      path: string
    ): Promise<Module<OutputType>>;
  }

  namespace vision {
    namespace ops {
      declare function nms<B extends TensorTypes, S extends TensorTypes>(
        boxes: Tensor<B>,
        scores: Tensor<S>,
        iouThreshold: number
      ): Tensor<B>;
    }

    namespace io {
      declare function readFile(filePath: string): Promise<Tensor<"uint8">>;
      declare function writeFile(
        data: Tensor<"uint8">,
        filePath: string
      ): Promise<void>;

      declare function readImage(filePath: string): Promise<Tensor<"uint8">>;

      // declare function encodePng(
      //   data: Tensor<"uint8">,
      //   compression: number
      // ): Promise<Tensor<"uint8">>;
      declare function encodeJpeg(
        data: Tensor<"uint8">,
        quality: number
      ): Promise<Tensor<"uint8">>;

      declare function decodeImage(
        rawData: Tensor<"uint8">
      ): Promise<Tensor<"uint8">>;
      declare function decodePng(data: Tensor<"uint8">): Promise<Tensor<"uint8">>;
      declare function decodeJpeg(
        data: Tensor<"uint8">
      ): Promise<Tensor<"uint8">>;
    }

  }

  namespace cuda {
    declare function isAvailable(): boolean;
    declare function deviceCount(): number;
  }

}

export = torch