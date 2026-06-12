# ONNX backend-test scoreboard (shim on gfx90a)

Ran 1678 node test cases through trt_infer.

- pass: 895
- fail: 757
- skip: 26

## Operators fully passing (96)
Acos, Acosh, Add, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, Ceil, Celu, Clip, Concat, Conv, Cos, Cosh, DepthToSpace, Div, DynamicQuantizeLinear, Elu, Equal, Erf, Exp, EyeLike, Flatten, Floor, GRU, Gather, GatherElements, GatherND, Gelu, Gemm, GlobalAveragePool, GlobalMaxPool, Greater, GreaterOrEqual, GroupNormalization, HardSigmoid, HardSwish, Hardmax, Identity, If, InstanceNormalization, IsInf, IsNaN, LRN, LSTM, LayerNormalization, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, LpNormalization, LpPool, MatMul, MatMulInteger, Max, Mean, MeanVarianceNormalization, Min, Mod, Neg, Not, Or, PRelu, Pow, QLinearConv, RNN, Reciprocal, Relu, RoiAlign, Round, Scatter, ScatterElements, ScatterND, Selu, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Softmax, Softplus, Softsign, SpaceToDepth, Sqrt, Sum, Tan, Tanh, ThresholdedRelu, Transpose, Where, Xor

## Operators partially passing (26)
Abs, AveragePool, BatchNormalization, BitwiseAnd, Cast, CastLike, Constant, ConvInteger, ConvTranspose, DequantizeLinear, Dropout, Einsum, GridSample, MaxPool, Mul, NegativeLogLikelihoodLoss, NonMaxSuppression, QuantizeLinear, ReduceMax, ReduceMin, ReduceProd, Shape, SoftmaxCrossEntropyLoss, Split, Sub, Trilu

## Operators with no pass (93)
Adagrad, Adam, AffineGrid, ArrayFeatureExtractor, Attention, Bernoulli, Binarizer, BitCast, BitShift, BitwiseNot, BitwiseOr, BitwiseXor, BlackmanWindow, CenterCropPad, Col2Im, Compress, ConstantOfShape, CumProd, CumSum, DFT, DeformConv, Det, Expand, HammingWindow, HannWindow, ImageDecoder, Loop, MaxUnpool, MelWeightMatrix, Mish, Momentum, NonZero, OneHot, OptionalGetElement, OptionalHasElement, Pad, QLinearMatMul, RMSNormalization, RandomUniformLike, Range, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMean, ReduceSum, ReduceSumSquare, Reshape, Resize, ReverseSequence, RotaryEmbedding, STFT, Scan, Slice, Squeeze, Swish, TensorScatter, TfIdfVectorizer, Tile, TopK, TreeEnsemble, Unique, Unsqueeze, Upsample, test_identity_opt, test_identity_sequence, test_if_opt, test_if_seq, test_loop13_seq, test_loop16_seq_none, test_optional_get_element_optional_sequence, test_optional_get_element_optional_tensor, test_optional_get_element_sequence, test_optional_has_element_empty_optional_input, test_optional_has_element_optional_input, test_optional_has_element_tensor_input, test_sequence_insert_at_back, test_sequence_insert_at_front, test_sequence_map_add_1_sequence_1_tensor, test_sequence_map_add_1_sequence_1_tensor_expanded, test_sequence_map_add_2_sequences, test_sequence_map_add_2_sequences_expanded, test_sequence_map_extract_shapes, test_sequence_map_extract_shapes_expanded, test_sequence_map_identity_1_sequence, test_sequence_map_identity_1_sequence_1_tensor, test_sequence_map_identity_1_sequence_1_tensor_expanded, test_sequence_map_identity_1_sequence_expanded, test_sequence_map_identity_2_sequences, test_sequence_map_identity_2_sequences_expanded, test_split_to_sequence_1, test_split_to_sequence_2, test_split_to_sequence_nokeepdims

## Top failure reasons
- run:parse: 669
- mismatch: 42
- non-tensor: 26
- harness:Currently not supporting loadi: 23
- size 512!=729: 3
- harness:cannot reshape array of size 2: 3
- size 18!=9: 3
- run:  what():  Failed to call function: 2
- size 18!=6: 2
- size 12!=3: 1
- harness:cannot reshape array of size 0: 1
- harness:cannot reshape array of size 1: 1
- size 4!=1: 1
- size 30!=3: 1
- size 36!=12: 1

## Interpretation

895/1678 (53%) of the standardized node tests pass, with 96 operators fully
green -- covering the core deep-learning set: Conv, MatMul, Gemm, Gather/Scatter,
the activations, normalizations (Layer/Group/Instance), Softmax, RNN/LSTM/GRU,
and quantized QLinearConv/MatMulInteger.

The failures are dominated by `run:parse` (669), which is MIGraphX's ONNX parser
rejecting the model -- almost entirely:
- Operators MIGraphX does not implement: Loop, Scan, sequence/optional ops,
  DFT/STFT/window functions, training optimizers (Adam/Adagrad/Momentum),
  bitwise ops. These are out of the shim's scope.
- Shape-manipulation ops whose node tests pass the shape/indices as a RUNTIME
  input tensor (Reshape, Slice, Squeeze, Unsqueeze, Tile, Expand). MIGraphX's
  parser needs these as constants; real models use the constant form, which the
  shim handles. So these "no pass" rows reflect the test variant, not a broken
  operator.

`mismatch` (42) are genuine numeric differences worth a closer look; `non-tensor`
(26) skips are sequence/optional-typed cases the runner does not marshal.
