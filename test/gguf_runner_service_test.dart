import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/services/gguf_runner_service.dart';

void main() {
  test('parses native GGUF inspection JSON', () {
    final inspection = GgufInspection.fromJson({
      'ok': true,
      'path': '/models/Qwen3-4B-Q8_0.gguf',
      'fileSizeBytes': 4280000000,
      'version': 3,
      'tensorCount': 3,
      'metadataCount': 12,
      'alignment': 32,
      'dataStartOffset': 4096,
      'parameterCount': 4000000000,
      'architecture': 'qwen2',
      'name': 'Qwen3 4B',
      'fileType': '7',
      'tokenizerModel': 'gpt2',
      'blockCount': 36,
      'contextLength': 32768,
      'embeddingLength': 2560,
      'feedForwardLength': 9728,
      'headCount': 32,
      'kvHeadCount': 8,
      'ropeFreqBase': '1000000',
      'tensorTypes': {'Q8_0': 2, 'F32': 1},
      'metadata': {'general.architecture': 'qwen2'},
      'tensors': [
        {
          'name': 'blk.0.attn_q.weight',
          'type': 'Q8_0',
          'typeId': 8,
          'relativeOffset': 0,
          'absoluteOffset': 4096,
          'elementCount': 6553600,
          'hasKnownByteSize': true,
          'byteSize': 6963200,
          'layerIndex': 0,
          'shape': [2560, 2560],
        },
      ],
    });

    expect(inspection.ok, isTrue);
    expect(inspection.architecture, 'qwen2');
    expect(inspection.blockCount, 36);
    expect(inspection.tensorTypes['Q8_0'], 2);
    expect(inspection.tensors.single.layerIndex, 0);
    expect(inspection.tensors.single.hasKnownByteSize, isTrue);
  });

  test('parses native layer streaming validation JSON', () {
    final validation = GgufLayerStreamingValidation.fromJson({
      'ok': true,
      'path': '/models/Qwen3-4B-Q8_0.gguf',
      'architecture': 'qwen2',
      'fileSizeBytes': 4280000000,
      'dataStartOffset': 4096,
      'blockCount': 2,
      'layersDiscovered': 2,
      'layersVisited': 2,
      'tensorsVisited': 18,
      'maxLayerSpanBytes': 150000000,
      'totalLayerSpanBytes': 300000000,
      'largestTensorBytes': 6963200,
      'missingLayerCount': 0,
      'missingLayers': [],
      'layers': [
        {
          'index': 0,
          'tensorCount': 9,
          'spanStart': 4096,
          'spanBytes': 150000000,
          'largestTensorBytes': 6963200,
          'mapped': true,
        },
        {
          'index': 1,
          'tensorCount': 9,
          'spanStart': 150004096,
          'spanBytes': 150000000,
          'largestTensorBytes': 6963200,
          'mapped': true,
        },
      ],
    });

    expect(validation.ok, isTrue);
    expect(validation.layersVisited, 2);
    expect(validation.missingLayers, isEmpty);
    expect(validation.layers.map((layer) => layer.index), [0, 1]);
    expect(validation.layers.first.mapped, isTrue);
  });

  test('parses native token embedding preview JSON', () {
    final embedding = GgufTokenEmbedding.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'tensorName': 'token_embd.weight',
      'tensorType': 'F32',
      'tensorTypeId': 0,
      'tokenId': 1,
      'embeddingLength': 3,
      'vocabSize': 2,
      'rowByteSize': 12,
      'requestedValueCount': 3,
      'returnedValueCount': 3,
      'min': 4.0,
      'max': 6.0,
      'mean': 5.0,
      'l2Norm': 8.774964,
      'checksum': 32.0,
      'shape': [3, 2],
      'values': [4.0, 5.0, 6.0],
    });

    expect(embedding.ok, isTrue);
    expect(embedding.tensorName, 'token_embd.weight');
    expect(embedding.tokenId, 1);
    expect(embedding.embeddingLength, 3);
    expect(embedding.values, [4.0, 5.0, 6.0]);
    expect(embedding.checksum, 32.0);
  });

  test('parses native RMSNorm preview JSON', () {
    final result = GgufRmsNormResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'tokenId': 1,
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'normTensorName': 'blk.0.attn_norm.weight',
      'normTensorType': 'F32',
      'embeddingLength': 3,
      'vocabSize': 2,
      'requestedValueCount': 3,
      'returnedValueCount': 3,
      'epsilon': 0.000001,
      'meanSquare': 25.6666667,
      'invRms': 0.1973855,
      'min': 0.789542,
      'max': 11.84313,
      'mean': 5.132025,
      'l2Norm': 13.06695,
      'checksum': 47.37252,
      'values': [0.789542, 3.94771, 11.84313],
    });

    expect(result.ok, isTrue);
    expect(result.tokenId, 1);
    expect(result.normTensorName, 'blk.0.attn_norm.weight');
    expect(result.embeddingLength, 3);
    expect(result.values, [0.789542, 3.94771, 11.84313]);
  });

  test('parses native MatVec preview JSON', () {
    final result = GgufMatVecResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'tokenId': 1,
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'normTensorName': 'blk.0.attn_norm.weight',
      'normTensorType': 'F32',
      'weightTensorName': 'blk.0.attn_q.weight',
      'weightTensorType': 'F32',
      'inputLength': 3,
      'outputLength': 2,
      'vocabSize': 2,
      'requestedValueCount': 2,
      'returnedValueCount': 2,
      'epsilon': 0.000001,
      'meanSquare': 25.6666667,
      'invRms': 0.1973855,
      'min': -2.763397,
      'max': 7.698034,
      'mean': 2.4673185,
      'l2Norm': 8.179084,
      'checksum': 12.632671,
      'weightShape': [3, 2],
      'values': [-2.763397, 7.698034],
    });

    expect(result.ok, isTrue);
    expect(result.weightTensorName, 'blk.0.attn_q.weight');
    expect(result.inputLength, 3);
    expect(result.outputLength, 2);
    expect(result.weightShape, [3, 2]);
    expect(result.values, [-2.763397, 7.698034]);
  });

  test('parses native RoPE MatVec preview JSON', () {
    final result = GgufRopeResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'tokenId': 1,
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'normTensorName': 'blk.0.attn_norm.weight',
      'normTensorType': 'F32',
      'weightTensorName': 'blk.0.attn_q.weight',
      'weightTensorType': 'F32',
      'inputLength': 3,
      'outputLength': 2,
      'vocabSize': 2,
      'position': 1,
      'headDim': 2,
      'headCount': 1,
      'rotatedPairCount': 1,
      'requestedValueCount': 2,
      'returnedValueCount': 2,
      'epsilon': 0.000001,
      'meanSquare': 25.6666667,
      'invRms': 0.1973855,
      'ropeTheta': 10000.0,
      'min': -7.963454,
      'max': 1.835814,
      'mean': -3.06382,
      'l2Norm': 8.172244,
      'checksum': -4.291826,
      'weightShape': [3, 2],
      'values': [-7.963454, 1.835814],
    });

    expect(result.ok, isTrue);
    expect(result.weightTensorName, 'blk.0.attn_q.weight');
    expect(result.position, 1);
    expect(result.headDim, 2);
    expect(result.headCount, 1);
    expect(result.rotatedPairCount, 1);
    expect(result.ropeTheta, 10000.0);
    expect(result.weightShape, [3, 2]);
    expect(result.values, [-7.963454, 1.835814]);
  });

  test('parses native KV cache preview JSON', () {
    final result = GgufKvCacheResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'normTensorName': 'blk.0.attn_norm.weight',
      'normTensorType': 'F32',
      'keyWeightTensorName': 'blk.0.attn_k.weight',
      'keyWeightTensorType': 'F32',
      'valueWeightTensorName': 'blk.0.attn_v.weight',
      'valueWeightTensorType': 'F32',
      'sequenceLength': 2,
      'embeddingLength': 3,
      'vocabSize': 2,
      'keyWidth': 2,
      'valueWidth': 2,
      'headDim': 2,
      'kvHeadCount': 1,
      'cacheElementCount': 8,
      'cacheBytesFp32': 32,
      'startPosition': 0,
      'readPosition': 1,
      'readAbsolutePosition': 1,
      'requestedValueCount': 2,
      'returnedKeyValueCount': 2,
      'returnedValueValueCount': 2,
      'epsilon': 0.000001,
      'lastMeanSquare': 25.6666667,
      'lastInvRms': 0.1973855,
      'ropeTheta': 10000.0,
      'keyMin': -2.1,
      'keyMax': 5.4,
      'keyMean': 1.6,
      'keyL2Norm': 8.1,
      'keyChecksum': 4.2,
      'valueMin': -1.1,
      'valueMax': 7.2,
      'valueMean': 3.1,
      'valueL2Norm': 9.4,
      'valueChecksum': 17.6,
      'tokenIds': [0, 1],
      'keyWeightShape': [3, 2],
      'valueWeightShape': [3, 2],
      'keyValues': [-2.1, 5.4],
      'valueValues': [3.5, 7.2],
    });

    expect(result.ok, isTrue);
    expect(result.sequenceLength, 2);
    expect(result.tokenIds, [0, 1]);
    expect(result.keyWeightTensorName, 'blk.0.attn_k.weight');
    expect(result.valueWeightTensorName, 'blk.0.attn_v.weight');
    expect(result.keyWidth, 2);
    expect(result.valueWidth, 2);
    expect(result.kvHeadCount, 1);
    expect(result.cacheBytesFp32, 32);
    expect(result.readPosition, 1);
    expect(result.keyValues, [-2.1, 5.4]);
    expect(result.valueValues, [3.5, 7.2]);
  });

  test('parses native single-head attention preview JSON', () {
    final result = GgufAttentionResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'queryTokenId': 1,
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'normTensorName': 'blk.0.attn_norm.weight',
      'normTensorType': 'F32',
      'queryWeightTensorName': 'blk.0.attn_q.weight',
      'queryWeightTensorType': 'F32',
      'keyWeightTensorName': 'blk.0.attn_k.weight',
      'keyWeightTensorType': 'F32',
      'valueWeightTensorName': 'blk.0.attn_v.weight',
      'valueWeightTensorType': 'F32',
      'sequenceLength': 2,
      'attentionLength': 2,
      'embeddingLength': 3,
      'vocabSize': 2,
      'queryWidth': 2,
      'keyWidth': 2,
      'valueWidth': 2,
      'headDim': 2,
      'headIndex': 0,
      'queryHeadCount': 1,
      'kvHeadCount': 1,
      'startPosition': 0,
      'queryPosition': 1,
      'readPosition': 1,
      'requestedValueCount': 2,
      'returnedOutputValueCount': 2,
      'returnedScoreValueCount': 2,
      'epsilon': 0.000001,
      'queryMeanSquare': 25.6666667,
      'queryInvRms': 0.1973855,
      'ropeTheta': 10000.0,
      'scale': 0.70710678,
      'maxScore': 9.1,
      'softmaxDenominator': 1.2,
      'outputMin': 3.0,
      'outputMax': 7.0,
      'outputMean': 5.0,
      'outputL2Norm': 7.6,
      'outputChecksum': 17.0,
      'tokenIds': [0, 1],
      'queryValues': [-7.9, 1.8],
      'attentionScores': [2.5, 9.1],
      'attentionProbabilities': [0.0014, 0.9986],
      'outputValues': [3.0, 7.0],
    });

    expect(result.ok, isTrue);
    expect(result.queryTokenId, 1);
    expect(result.queryWeightTensorName, 'blk.0.attn_q.weight');
    expect(result.keyWeightTensorName, 'blk.0.attn_k.weight');
    expect(result.valueWeightTensorName, 'blk.0.attn_v.weight');
    expect(result.attentionLength, 2);
    expect(result.headDim, 2);
    expect(result.headIndex, 0);
    expect(result.queryValues, [-7.9, 1.8]);
    expect(result.attentionScores, [2.5, 9.1]);
    expect(result.attentionProbabilities, [0.0014, 0.9986]);
    expect(result.outputValues, [3.0, 7.0]);
  });

  test('parses native multi-head attention preview JSON', () {
    final result = GgufMultiHeadAttentionResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'queryTokenId': 1,
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'normTensorName': 'blk.0.attn_norm.weight',
      'normTensorType': 'F32',
      'queryWeightTensorName': 'blk.0.attn_q_multi.weight',
      'queryWeightTensorType': 'F32',
      'keyWeightTensorName': 'blk.0.attn_k.weight',
      'keyWeightTensorType': 'F32',
      'valueWeightTensorName': 'blk.0.attn_v.weight',
      'valueWeightTensorType': 'F32',
      'sequenceLength': 2,
      'attentionLength': 2,
      'embeddingLength': 3,
      'vocabSize': 2,
      'queryWidth': 4,
      'keyWidth': 2,
      'valueWidth': 2,
      'headDim': 2,
      'queryHeadCount': 2,
      'kvHeadCount': 1,
      'groupSize': 2,
      'startPosition': 0,
      'queryPosition': 1,
      'readPosition': 1,
      'requestedValueCount': 4,
      'returnedOutputValueCount': 4,
      'returnedScoreValueCount': 2,
      'epsilon': 0.000001,
      'queryMeanSquare': 25.6666667,
      'queryInvRms': 0.1973855,
      'ropeTheta': 10000.0,
      'scale': 0.70710678,
      'outputMin': 3.0,
      'outputMax': 7.0,
      'outputMean': 5.0,
      'outputL2Norm': 10.2,
      'outputChecksum': 31.0,
      'tokenIds': [0, 1],
      'qHeadToKvHead': [0, 0],
      'queryValues': [-7.9, 1.8, 2.2, -0.4],
      'firstHeadScores': [2.5, 9.1],
      'firstHeadProbabilities': [0.0014, 0.9986],
      'outputValues': [3.0, 7.0, 4.0, 6.0],
    });

    expect(result.ok, isTrue);
    expect(result.queryTokenId, 1);
    expect(result.queryWeightTensorName, 'blk.0.attn_q_multi.weight');
    expect(result.queryWidth, 4);
    expect(result.headDim, 2);
    expect(result.queryHeadCount, 2);
    expect(result.kvHeadCount, 1);
    expect(result.groupSize, 2);
    expect(result.qHeadToKvHead, [0, 0]);
    expect(result.queryValues, [-7.9, 1.8, 2.2, -0.4]);
    expect(result.firstHeadScores, [2.5, 9.1]);
    expect(result.firstHeadProbabilities, [0.0014, 0.9986]);
    expect(result.outputValues, [3.0, 7.0, 4.0, 6.0]);
  });

  test('parses native transformer layer preview JSON', () {
    final result = GgufTransformerLayerResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'queryTokenId': 1,
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'attnNormTensorName': 'blk.0.attn_norm.weight',
      'attnNormTensorType': 'F32',
      'queryWeightTensorName': 'blk.0.attn_q_multi.weight',
      'queryWeightTensorType': 'F32',
      'keyWeightTensorName': 'blk.0.attn_k.weight',
      'keyWeightTensorType': 'F32',
      'valueWeightTensorName': 'blk.0.attn_v.weight',
      'valueWeightTensorType': 'F32',
      'outputWeightTensorName': 'blk.0.attn_o.weight',
      'outputWeightTensorType': 'F32',
      'ffnNormTensorName': 'blk.0.ffn_norm.weight',
      'ffnNormTensorType': 'F32',
      'gateWeightTensorName': 'blk.0.ffn_gate.weight',
      'gateWeightTensorType': 'F32',
      'upWeightTensorName': 'blk.0.ffn_up.weight',
      'upWeightTensorType': 'F32',
      'downWeightTensorName': 'blk.0.ffn_down.weight',
      'downWeightTensorType': 'F32',
      'sequenceLength': 2,
      'attentionLength': 2,
      'embeddingLength': 3,
      'vocabSize': 2,
      'queryWidth': 4,
      'keyWidth': 2,
      'valueWidth': 2,
      'attentionOutputWidth': 4,
      'hiddenSize': 3,
      'ffnWidth': 4,
      'headDim': 2,
      'queryHeadCount': 2,
      'kvHeadCount': 1,
      'groupSize': 2,
      'startPosition': 0,
      'queryPosition': 1,
      'readPosition': 1,
      'requestedValueCount': 4,
      'returnedOutputValueCount': 3,
      'returnedAttentionValueCount': 4,
      'returnedFfnValueCount': 4,
      'returnedScoreValueCount': 2,
      'epsilon': 0.000001,
      'attnMeanSquare': 25.6666667,
      'attnInvRms': 0.1973855,
      'ffnMeanSquare': 30.5,
      'ffnInvRms': 0.181071,
      'ropeTheta': 10000.0,
      'scale': 0.70710678,
      'outputMin': 4.0,
      'outputMax': 9.0,
      'outputMean': 6.5,
      'outputL2Norm': 11.2,
      'outputChecksum': 38.0,
      'tokenIds': [0, 1],
      'qHeadToKvHead': [0, 0],
      'queryValues': [-7.9, 1.8, 2.2, -0.4],
      'attentionValues': [3.0, 7.0, 4.0, 6.0],
      'attentionProjectedValues': [1.0, 2.0, 3.0],
      'postAttentionValues': [5.0, 7.0, 9.0],
      'gateValues': [0.5, 1.5, -0.25, 2.0],
      'upValues': [1.0, -0.5, 0.75, 2.5],
      'ffnHiddenValues': [0.31, -0.61, -0.08, 4.4],
      'ffnOutputValues': [0.25, -0.5, 1.0],
      'outputValues': [5.25, 6.5, 10.0],
      'firstHeadScores': [2.5, 9.1],
      'firstHeadProbabilities': [0.0014, 0.9986],
    });

    expect(result.ok, isTrue);
    expect(result.queryTokenId, 1);
    expect(result.attnNormTensorName, 'blk.0.attn_norm.weight');
    expect(result.outputWeightTensorName, 'blk.0.attn_o.weight');
    expect(result.ffnNormTensorName, 'blk.0.ffn_norm.weight');
    expect(result.gateWeightTensorName, 'blk.0.ffn_gate.weight');
    expect(result.upWeightTensorName, 'blk.0.ffn_up.weight');
    expect(result.downWeightTensorName, 'blk.0.ffn_down.weight');
    expect(result.hiddenSize, 3);
    expect(result.ffnWidth, 4);
    expect(result.groupSize, 2);
    expect(result.qHeadToKvHead, [0, 0]);
    expect(result.attentionValues, [3.0, 7.0, 4.0, 6.0]);
    expect(result.postAttentionValues, [5.0, 7.0, 9.0]);
    expect(result.ffnHiddenValues, [0.31, -0.61, -0.08, 4.4]);
    expect(result.outputValues, [5.25, 6.5, 10.0]);
  });

  test('parses native transformer stack preview JSON', () {
    final result = GgufTransformerStackResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'attnNormTensorSuffix': 'attn_norm.weight',
      'queryWeightTensorSuffix': 'attn_q_multi.weight',
      'keyWeightTensorSuffix': 'attn_k.weight',
      'valueWeightTensorSuffix': 'attn_v.weight',
      'outputWeightTensorSuffix': 'attn_o.weight',
      'ffnNormTensorSuffix': 'ffn_norm.weight',
      'gateWeightTensorSuffix': 'ffn_gate.weight',
      'upWeightTensorSuffix': 'ffn_up.weight',
      'downWeightTensorSuffix': 'ffn_down.weight',
      'blockCount': 2,
      'layerCount': 2,
      'layersVisited': 2,
      'tensorsVisited': 18,
      'sequenceLength': 2,
      'readPosition': 1,
      'selectedTokenId': 1,
      'startPosition': 0,
      'hiddenSize': 3,
      'vocabSize': 2,
      'queryWidth': 4,
      'keyWidth': 2,
      'valueWidth': 2,
      'ffnWidth': 4,
      'headDim': 2,
      'queryHeadCount': 2,
      'kvHeadCount': 1,
      'groupSize': 2,
      'kvCacheLayerCount': 2,
      'kvCacheElementCount': 16,
      'kvCacheBytesFp32': 64,
      'requestedValueCount': 4,
      'returnedOutputValueCount': 3,
      'returnedAttentionValueCount': 4,
      'returnedFfnValueCount': 4,
      'returnedScoreValueCount': 2,
      'epsilon': 0.000001,
      'ropeTheta': 10000.0,
      'scale': 0.70710678,
      'lastLayerAttnMeanSquare': 12.5,
      'lastLayerAttnInvRms': 0.2828,
      'lastLayerFfnMeanSquare': 20.5,
      'lastLayerFfnInvRms': 0.2208,
      'outputMin': 1.0,
      'outputMax': 3.0,
      'outputMean': 2.0,
      'outputL2Norm': 3.74,
      'outputChecksum': 14.0,
      'tokenIds': [0, 1],
      'qHeadToKvHead': [0, 0],
      'layerOutputChecksums': [10.0, 14.0],
      'lastLayerQueryValues': [0.1, 0.2, 0.3, 0.4],
      'lastLayerAttentionValues': [1.0, 2.0, 3.0, 4.0],
      'lastLayerAttentionProjectedValues': [0.5, 0.75, 1.0],
      'lastLayerPostAttentionValues': [1.5, 2.75, 4.0],
      'lastLayerGateValues': [0.2, -0.1, 0.4, 0.7],
      'lastLayerUpValues': [1.0, 1.5, -0.5, 2.0],
      'lastLayerFfnHiddenValues': [0.11, -0.07, -0.12, 0.94],
      'lastLayerFfnOutputValues': [0.25, 0.5, -0.25],
      'outputValues': [1.75, 3.25, 3.75],
      'lastLayerFirstHeadScores': [0.9, 1.2],
      'lastLayerFirstHeadProbabilities': [0.425, 0.575],
    });

    expect(result.ok, isTrue);
    expect(result.layerCount, 2);
    expect(result.layersVisited, 2);
    expect(result.tensorsVisited, 18);
    expect(result.selectedTokenId, 1);
    expect(result.queryWeightTensorSuffix, 'attn_q_multi.weight');
    expect(result.hiddenSize, 3);
    expect(result.queryHeadCount, 2);
    expect(result.groupSize, 2);
    expect(result.kvCacheLayerCount, 2);
    expect(result.kvCacheBytesFp32, 64);
    expect(result.layerOutputChecksums, [10.0, 14.0]);
    expect(result.lastLayerAttentionValues, [1.0, 2.0, 3.0, 4.0]);
    expect(result.outputValues, [1.75, 3.25, 3.75]);
    expect(result.lastLayerFirstHeadProbabilities, [0.425, 0.575]);
  });

  test('parses native next-token preview JSON', () {
    final result = GgufNextTokenResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'finalNormTensorName': 'output_norm.weight',
      'finalNormTensorType': 'F32',
      'lmHeadTensorName': 'output.weight',
      'lmHeadTensorType': 'F32',
      'attnNormTensorSuffix': 'attn_norm.weight',
      'queryWeightTensorSuffix': 'attn_q_multi.weight',
      'keyWeightTensorSuffix': 'attn_k.weight',
      'valueWeightTensorSuffix': 'attn_v.weight',
      'outputWeightTensorSuffix': 'attn_o.weight',
      'ffnNormTensorSuffix': 'ffn_norm.weight',
      'gateWeightTensorSuffix': 'ffn_gate.weight',
      'upWeightTensorSuffix': 'ffn_up.weight',
      'downWeightTensorSuffix': 'ffn_down.weight',
      'blockCount': 2,
      'layerCount': 2,
      'layersVisited': 2,
      'tensorsVisited': 20,
      'sequenceLength': 2,
      'readPosition': 1,
      'selectedTokenId': 1,
      'nextTokenId': 3,
      'startPosition': 0,
      'hiddenSize': 3,
      'vocabSize': 2,
      'lmHeadInputLength': 3,
      'logitCount': 4,
      'headDim': 2,
      'queryHeadCount': 2,
      'kvHeadCount': 1,
      'groupSize': 2,
      'kvCacheLayerCount': 2,
      'kvCacheElementCount': 16,
      'kvCacheBytesFp32': 64,
      'requestedTopK': 3,
      'returnedTopK': 3,
      'requestedValueCount': 4,
      'returnedHiddenValueCount': 3,
      'returnedNormalizedValueCount': 3,
      'returnedLogitValueCount': 4,
      'epsilon': 0.000001,
      'ropeTheta': 10000.0,
      'scale': 0.70710678,
      'finalMeanSquare': 14.5,
      'finalInvRms': 0.2626,
      'nextTokenLogit': 2.75,
      'hiddenMin': 1.0,
      'hiddenMax': 3.0,
      'hiddenMean': 2.0,
      'hiddenL2Norm': 3.74,
      'hiddenChecksum': 14.0,
      'normalizedMin': 0.5,
      'normalizedMax': 1.5,
      'normalizedMean': 1.0,
      'normalizedL2Norm': 1.87,
      'normalizedChecksum': 7.0,
      'logitMin': -1.0,
      'logitMax': 2.75,
      'logitMean': 0.75,
      'logitL2Norm': 3.0,
      'logitChecksum': 8.5,
      'tokenIds': [0, 1],
      'qHeadToKvHead': [0, 0],
      'layerOutputChecksums': [10.0, 14.0],
      'hiddenValues': [1.0, 2.0, 3.0],
      'normalizedValues': [0.5, 1.0, 1.5],
      'logitValues': [0.25, -1.0, 1.25, 2.75],
      'topTokenIds': [3, 2, 0],
      'topLogits': [2.75, 1.25, 0.25],
    });

    expect(result.ok, isTrue);
    expect(result.finalNormTensorName, 'output_norm.weight');
    expect(result.lmHeadTensorName, 'output.weight');
    expect(result.nextTokenId, 3);
    expect(result.logitCount, 4);
    expect(result.topTokenIds, [3, 2, 0]);
    expect(result.topLogits, [2.75, 1.25, 0.25]);
    expect(result.normalizedValues, [0.5, 1.0, 1.5]);
  });

  test('parses native greedy generation preview JSON', () {
    final result = GgufGeneratedTokensResult.fromJson({
      'ok': true,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'embeddingTensorName': 'token_embd.weight',
      'embeddingTensorType': 'F32',
      'finalNormTensorName': 'output_norm.weight',
      'finalNormTensorType': 'F32',
      'lmHeadTensorName': 'output.weight',
      'lmHeadTensorType': 'F32',
      'cacheReused': true,
      'blockCount': 2,
      'layerCount': 2,
      'layersVisited': 6,
      'tensorsVisited': 60,
      'sequenceLength': 2,
      'generatedTokenCount': 3,
      'totalTokenCount': 5,
      'maxNewTokens': 3,
      'readPosition': 1,
      'selectedTokenId': 1,
      'firstGeneratedTokenId': 0,
      'lastGeneratedTokenId': 1,
      'startPosition': 0,
      'hiddenSize': 3,
      'vocabSize': 2,
      'lmHeadInputLength': 3,
      'logitCount': 2,
      'headDim': 2,
      'queryHeadCount': 2,
      'kvHeadCount': 1,
      'groupSize': 2,
      'kvCacheLayerCount': 2,
      'initialKvCacheTokenCount': 2,
      'finalKvCacheTokenCount': 4,
      'initialKvCacheElementCount': 16,
      'finalKvCacheElementCount': 32,
      'initialKvCacheBytesFp32': 64,
      'finalKvCacheBytesFp32': 128,
      'requestedTopK': 2,
      'returnedTopK': 2,
      'requestedValueCount': 4,
      'returnedHiddenValueCount': 3,
      'returnedNormalizedValueCount': 3,
      'returnedLogitValueCount': 2,
      'epsilon': 0.000001,
      'ropeTheta': 10000.0,
      'scale': 0.70710678,
      'finalMeanSquare': 14.5,
      'finalInvRms': 0.2626,
      'lastTokenLogit': 2.75,
      'hiddenMin': 1.0,
      'hiddenMax': 3.0,
      'hiddenMean': 2.0,
      'hiddenL2Norm': 3.74,
      'hiddenChecksum': 14.0,
      'normalizedMin': 0.5,
      'normalizedMax': 1.5,
      'normalizedMean': 1.0,
      'normalizedL2Norm': 1.87,
      'normalizedChecksum': 7.0,
      'logitMin': -1.0,
      'logitMax': 2.75,
      'logitMean': 0.75,
      'logitL2Norm': 3.0,
      'logitChecksum': 4.5,
      'tokenIds': [0, 1],
      'generatedTokenIds': [0, 0, 1],
      'allTokenIds': [0, 1, 0, 0, 1],
      'generatedTokenLogits': [1.25, 1.5, 2.75],
      'qHeadToKvHead': [0, 0],
      'layerOutputChecksums': [10.0, 14.0],
      'hiddenValues': [1.0, 2.0, 3.0],
      'normalizedValues': [0.5, 1.0, 1.5],
      'logitValues': [0.25, 2.75],
      'topTokenIds': [1, 0],
      'topLogits': [2.75, 0.25],
    });

    expect(result.ok, isTrue);
    expect(result.cacheReused, isTrue);
    expect(result.generatedTokenCount, 3);
    expect(result.totalTokenCount, 5);
    expect(result.tensorsVisited, 60);
    expect(result.generatedTokenIds, [0, 0, 1]);
    expect(result.allTokenIds, [0, 1, 0, 0, 1]);
    expect(result.finalKvCacheTokenCount, 4);
    expect(result.topTokenIds, [1, 0]);
  });

  test('parses native persistent session create JSON', () {
    final result = GgufSessionCreateResult.fromJson({
      'ok': true,
      'sessionId': 42,
      'activeSessionCount': 1,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'prefilled': true,
      'blockCount': 2,
      'layerCount': 2,
      'promptTokenCount': 2,
      'totalTokenCount': 2,
      'generatedTokenCount': 0,
      'startPosition': 0,
      'nextPosition': 2,
      'hiddenSize': 3,
      'vocabSize': 2,
      'headDim': 2,
      'queryHeadCount': 2,
      'kvHeadCount': 1,
      'groupSize': 2,
      'kvCacheLayerCount': 2,
      'kvCacheTokenCount': 2,
      'kvCacheElementCount': 16,
      'kvCacheBytesFp32': 64,
      'requestedValueCount': 4,
      'returnedHiddenValueCount': 3,
      'epsilon': 0.000001,
      'ropeTheta': 10000.0,
      'hiddenMin': 1.0,
      'hiddenMax': 3.0,
      'hiddenMean': 2.0,
      'hiddenL2Norm': 3.74,
      'hiddenChecksum': 14.0,
      'tokenIds': [0, 1],
      'allTokenIds': [0, 1],
      'qHeadToKvHead': [0, 0],
      'layerOutputChecksums': [10.0, 14.0],
      'hiddenValues': [1.0, 2.0, 3.0],
    });

    expect(result.ok, isTrue);
    expect(result.sessionId, 42);
    expect(result.prefilled, isTrue);
    expect(result.nextPosition, 2);
    expect(result.kvCacheTokenCount, 2);
    expect(result.tokenIds, [0, 1]);
    expect(result.hiddenValues, [1.0, 2.0, 3.0]);
  });

  test('parses native persistent session decode JSON', () {
    final result = GgufSessionDecodeResult.fromJson({
      'ok': true,
      'sessionId': 42,
      'path': '/models/tiny.gguf',
      'architecture': 'qwen2',
      'decodedPosition': 2,
      'nextPosition': 3,
      'generatedTokenId': 1,
      'generatedTokenCount': 1,
      'totalTokenCount': 3,
      'hiddenSize': 3,
      'vocabSize': 2,
      'logitCount': 2,
      'kvCacheTokenCount': 3,
      'kvCacheElementCount': 24,
      'kvCacheBytesFp32': 96,
      'requestedTopK': 2,
      'returnedTopK': 2,
      'requestedValueCount': 4,
      'returnedHiddenValueCount': 3,
      'returnedNormalizedValueCount': 3,
      'returnedLogitValueCount': 2,
      'generatedTokenLogit': 2.75,
      'epsilon': 0.000001,
      'ropeTheta': 10000.0,
      'finalMeanSquare': 14.5,
      'finalInvRms': 0.2626,
      'hiddenMin': 1.0,
      'hiddenMax': 3.0,
      'hiddenMean': 2.0,
      'hiddenL2Norm': 3.74,
      'hiddenChecksum': 14.0,
      'normalizedMin': 0.5,
      'normalizedMax': 1.5,
      'normalizedMean': 1.0,
      'normalizedL2Norm': 1.87,
      'normalizedChecksum': 7.0,
      'logitMin': -1.0,
      'logitMax': 2.75,
      'logitMean': 0.75,
      'logitL2Norm': 3.0,
      'logitChecksum': 4.5,
      'generatedTokenIds': [1],
      'allTokenIds': [0, 1, 1],
      'generatedTokenLogits': [2.75],
      'hiddenValues': [1.0, 2.0, 3.0],
      'normalizedValues': [0.5, 1.0, 1.5],
      'logitValues': [0.25, 2.75],
      'topTokenIds': [1, 0],
      'topLogits': [2.75, 0.25],
    });

    expect(result.ok, isTrue);
    expect(result.sessionId, 42);
    expect(result.generatedTokenId, 1);
    expect(result.generatedTokenCount, 1);
    expect(result.allTokenIds, [0, 1, 1]);
    expect(result.topTokenIds, [1, 0]);
  });

  test('parses native persistent session close JSON', () {
    final result = GgufSessionCloseResult.fromJson({
      'ok': true,
      'sessionId': 42,
      'closed': true,
      'activeSessionCount': 0,
    });

    expect(result.ok, isTrue);
    expect(result.sessionId, 42);
    expect(result.closed, isTrue);
    expect(result.activeSessionCount, 0);
  });

  test('parses native error JSON without throwing in factory', () {
    final inspection = GgufInspection.fromJson({
      'ok': false,
      'error': 'File is not a GGUF model',
    });

    expect(inspection.ok, isFalse);
    expect(inspection.error, 'File is not a GGUF model');
  });
}
