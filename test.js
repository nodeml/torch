//C++ Tensor Slicing https://stackoverflow.com/questions/56908893/copy-a-chunk-of-one-tensor-into-another-one-in-c-api?rq=4
const torch = require('./lib')

// let a = torch.tensor(new Int32Array([1, 2, 3, 4, 5, 6])).reshape([2, 3]);
// let b = torch
//   .tensor(new Int32Array([1, 2, 3, 4, 5, 6].reverse()))
//   .reshape([2, 3]);
// console.log(a.toMultiArray());
// // a = a.add(b);
// a.set(b.get(1, [0, 3]), 1, [0, 3]);
// console.log(a.toMultiArray());
// // torch.jit
// //   .load("./segmentation.torchscript")
// //   .forward(torch.rand([1, 3, 640, 640]))
// //   .then((h) => {
// //     console.log("FORWARD RESULT", h);
// //   });
// console.log(
//   a.get(1, [0, 3]).toMultiArray(),
//   torch.arange(0, 20, torch.types.double).toArray(),
//   torch.arange(0, 20, torch.types.uint8).toArray(),
//   a.amax(0).toMultiArray(),
//   b.toMultiArray(),
//   torch.cat([a, b]).toMultiArray(),
//   torch.equal(a, b).toMultiArray()
// );

// let g = torch.zeros([4, 3]);

// "binary": {
//   "napi_versions": [
//     7
//   ]
// }

// const bytes = torch.vision.io.readFile("./test.jpg");
// let img = torch.vision.io.decodeImage(bytes);
// img = torch.nn.functional.interpolate(
//   torch.nn.functional.pad(img, [0, Math.floor(1400 - 994)]).unsqueeze(0),
//   [640, 640]
// );
const formatMemoryUsage = (data) => `${Math.round(data / 1024 / 1024 * 100) / 100} MB`;
// console.log(img.shape, img.dtype);
// const x = torch.tensor([true,false,true])
// console.log(x.dtype,x.get([null,2]).toArray())
async function main() {
    
    let x = torch.arange(1000)
    const y = 1
    // x.set([0,null],torch.tensor([1,1,1,1,1,1])  )
    const start = formatMemoryUsage(process.memoryUsage().heapUsed)
    // for(let i = 0; i < 10000000; i++){
    //     x = x.add(y)
    //     console.log(start,formatMemoryUsage(process.memoryUsage().heapUsed))
    // }
    
}

main()

