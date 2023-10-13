const torch = require("./lib");
const imageExample = torch.rand([3, 200, 500]);
console.log(imageExample.shape);
console.log(
  torch.nn.functional
    .interpolate(imageExample.unsqueeze(0), [300, 1000])
    .squeeze(0).dtype
);
