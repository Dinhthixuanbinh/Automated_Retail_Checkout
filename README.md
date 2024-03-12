## Introduction
By having an image and corresponding bounding box. The model will perform slicing of the object and create vector embedding and comparison in the database to recognize the product name.

## Performance
Evaluations on val data
| Method              | Input Size        | Easy  | Medium | Hard  | Average  | 
| ------------------- | --------------- | ----- | ------ | ----- | ----- |
| EfficientNet-B0     | 128              | 73.26%| 70.34% | 69.94%| 71.18%   |
| EfficientNet-B0     | 256              | 83.58%| 80.93% | 79.75%| 81.42%   |
| EfficientNet-B0     | 512              | 85.71%| 83.10% | 82.63%| 83.82%   |
| EfficientNet-B2     | 256              | 82.62%| 81.09% | 78.5% | 80.73%   |
| **EfficientNet-B2**    | **512**              | 86.88%| 85.05% | 83.51%| 85.14%   |

Evaluation on test data 
| Clutter mode | Methods           | Input size | Accuracy |
|--------------|-------------------|------------|----------|
| Easy         | EfficientNet-B2  | 512        | 86.21%   |
| Medium       | EfficientNet-B2  | 512        | 84.74%   |
| Hard         | EfficientNet-B2  | 512        | 83.50%   |
| Average      | EfficientNet-B2  | 512        | 84.81%   |

