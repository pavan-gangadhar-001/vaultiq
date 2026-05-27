import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:vaultiq/src/services/gguf_runner_service.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('native GGUF runner inspects and maps layer spans', (
    tester,
  ) async {
    final directory = await getApplicationDocumentsDirectory();
    final file = File(p.join(directory.path, 'tiny_layer_streaming.gguf'));
    await file.writeAsBytes(_tinyGgufFixture(), flush: true);

    final service = GgufRunnerService();
    final inspection = await service.inspectGguf(file.path);
    final validation = await service.validateLayerStreaming(file.path);
    final tokenEmbedding = await service.readTokenEmbedding(
      path: file.path,
      tokenId: 1,
      maxValues: 3,
    );
    final rmsNorm = await service.rmsNormTokenEmbedding(
      path: file.path,
      tokenId: 1,
      normTensorName: 'blk.0.attn_norm.weight',
      epsilon: 0.000001,
      maxValues: 3,
    );
    final matVec = await service.matVecRmsNormTokenEmbedding(
      path: file.path,
      tokenId: 1,
      normTensorName: 'blk.0.attn_norm.weight',
      weightTensorName: 'blk.0.attn_q.weight',
      epsilon: 0.000001,
      maxValues: 2,
    );
    final rope = await service.ropeMatVecRmsNormTokenEmbedding(
      path: file.path,
      tokenId: 1,
      normTensorName: 'blk.0.attn_norm.weight',
      weightTensorName: 'blk.0.attn_q.weight',
      epsilon: 0.000001,
      position: 1,
      headDim: 2,
      ropeTheta: 10000,
      maxValues: 2,
    );
    final kvCache = await service.kvCacheRmsNormTokenEmbeddings(
      path: file.path,
      tokenIds: [0, 1],
      normTensorName: 'blk.0.attn_norm.weight',
      keyWeightTensorName: 'blk.0.attn_k.weight',
      valueWeightTensorName: 'blk.0.attn_v.weight',
      epsilon: 0.000001,
      startPosition: 0,
      readPosition: 1,
      headDim: 2,
      ropeTheta: 10000,
      maxValues: 2,
    );
    final attention = await service.attentionSingleHead(
      path: file.path,
      tokenIds: [0, 1],
      queryTokenId: 1,
      normTensorName: 'blk.0.attn_norm.weight',
      queryWeightTensorName: 'blk.0.attn_q.weight',
      keyWeightTensorName: 'blk.0.attn_k.weight',
      valueWeightTensorName: 'blk.0.attn_v.weight',
      epsilon: 0.000001,
      startPosition: 0,
      queryPosition: 1,
      readPosition: 1,
      attentionLength: 2,
      headDim: 2,
      headIndex: 0,
      ropeTheta: 10000,
      maxValues: 2,
    );
    final multiHeadAttention = await service.attentionMultiHead(
      path: file.path,
      tokenIds: [0, 1],
      queryTokenId: 1,
      normTensorName: 'blk.0.attn_norm.weight',
      queryWeightTensorName: 'blk.0.attn_q_multi.weight',
      keyWeightTensorName: 'blk.0.attn_k.weight',
      valueWeightTensorName: 'blk.0.attn_v.weight',
      epsilon: 0.000001,
      startPosition: 0,
      queryPosition: 1,
      readPosition: 1,
      attentionLength: 2,
      headDim: 2,
      ropeTheta: 10000,
      maxValues: 4,
    );
    final layer = await service.transformerLayer(
      path: file.path,
      tokenIds: [0, 1],
      queryTokenId: 1,
      attnNormTensorName: 'blk.0.attn_norm.weight',
      queryWeightTensorName: 'blk.0.attn_q_multi.weight',
      keyWeightTensorName: 'blk.0.attn_k.weight',
      valueWeightTensorName: 'blk.0.attn_v.weight',
      outputWeightTensorName: 'blk.0.attn_o.weight',
      ffnNormTensorName: 'blk.0.ffn_norm.weight',
      gateWeightTensorName: 'blk.0.ffn_gate.weight',
      upWeightTensorName: 'blk.0.ffn_up.weight',
      downWeightTensorName: 'blk.0.ffn_down.weight',
      epsilon: 0.000001,
      startPosition: 0,
      queryPosition: 1,
      readPosition: 1,
      attentionLength: 2,
      headDim: 2,
      ropeTheta: 10000,
      maxValues: 4,
    );
    final stack = await service.transformerStack(
      path: file.path,
      tokenIds: [0, 1],
      attnNormTensorSuffix: 'attn_norm.weight',
      queryWeightTensorSuffix: 'attn_q_multi.weight',
      keyWeightTensorSuffix: 'attn_k.weight',
      valueWeightTensorSuffix: 'attn_v.weight',
      outputWeightTensorSuffix: 'attn_o.weight',
      ffnNormTensorSuffix: 'ffn_norm.weight',
      gateWeightTensorSuffix: 'ffn_gate.weight',
      upWeightTensorSuffix: 'ffn_up.weight',
      downWeightTensorSuffix: 'ffn_down.weight',
      epsilon: 0.000001,
      startPosition: 0,
      readPosition: 1,
      layerCount: 2,
      headDim: 2,
      ropeTheta: 10000,
      maxValues: 4,
    );
    final nextToken = await service.generateNextToken(
      path: file.path,
      tokenIds: [0, 1],
      attnNormTensorSuffix: 'attn_norm.weight',
      queryWeightTensorSuffix: 'attn_q_multi.weight',
      keyWeightTensorSuffix: 'attn_k.weight',
      valueWeightTensorSuffix: 'attn_v.weight',
      outputWeightTensorSuffix: 'attn_o.weight',
      ffnNormTensorSuffix: 'ffn_norm.weight',
      gateWeightTensorSuffix: 'ffn_gate.weight',
      upWeightTensorSuffix: 'ffn_up.weight',
      downWeightTensorSuffix: 'ffn_down.weight',
      finalNormTensorName: 'output_norm.weight',
      lmHeadTensorName: 'output.weight',
      epsilon: 0.000001,
      startPosition: 0,
      readPosition: 1,
      layerCount: 2,
      headDim: 2,
      ropeTheta: 10000,
      topK: 2,
      maxValues: 4,
    );
    final generated = await service.generateTokens(
      path: file.path,
      tokenIds: [0, 1],
      attnNormTensorSuffix: 'attn_norm.weight',
      queryWeightTensorSuffix: 'attn_q_multi.weight',
      keyWeightTensorSuffix: 'attn_k.weight',
      valueWeightTensorSuffix: 'attn_v.weight',
      outputWeightTensorSuffix: 'attn_o.weight',
      ffnNormTensorSuffix: 'ffn_norm.weight',
      gateWeightTensorSuffix: 'ffn_gate.weight',
      upWeightTensorSuffix: 'ffn_up.weight',
      downWeightTensorSuffix: 'ffn_down.weight',
      finalNormTensorName: 'output_norm.weight',
      lmHeadTensorName: 'output.weight',
      epsilon: 0.000001,
      startPosition: 0,
      readPosition: 1,
      layerCount: 2,
      headDim: 2,
      ropeTheta: 10000,
      maxNewTokens: 3,
      topK: 2,
      maxValues: 4,
    );
    final session = await service.createSession(
      path: file.path,
      tokenIds: [0, 1],
      attnNormTensorSuffix: 'attn_norm.weight',
      queryWeightTensorSuffix: 'attn_q_multi.weight',
      keyWeightTensorSuffix: 'attn_k.weight',
      valueWeightTensorSuffix: 'attn_v.weight',
      outputWeightTensorSuffix: 'attn_o.weight',
      ffnNormTensorSuffix: 'ffn_norm.weight',
      gateWeightTensorSuffix: 'ffn_gate.weight',
      upWeightTensorSuffix: 'ffn_up.weight',
      downWeightTensorSuffix: 'ffn_down.weight',
      finalNormTensorName: 'output_norm.weight',
      lmHeadTensorName: 'output.weight',
      epsilon: 0.000001,
      startPosition: 0,
      readPosition: 1,
      layerCount: 2,
      headDim: 2,
      ropeTheta: 10000,
      maxValues: 4,
    );
    final sessionDecodes = <GgufSessionDecodeResult>[];
    late final GgufSessionCloseResult closedSession;
    try {
      for (var i = 0; i < 3; i++) {
        sessionDecodes.add(
          await service.decodeSession(
            sessionId: session.sessionId,
            topK: 2,
            maxValues: 4,
          ),
        );
      }
    } finally {
      closedSession = await service.closeSession(session.sessionId);
    }

    expect(inspection.ok, isTrue);
    expect(inspection.architecture, 'qwen2');
    expect(inspection.blockCount, 2);
    expect(inspection.tensorCount, 22);
    expect(inspection.tensors.map((tensor) => tensor.layerIndex), [
      -1,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      -1,
      -1,
    ]);

    expect(validation.ok, isTrue);
    expect(validation.blockCount, 2);
    expect(validation.layersVisited, 2);
    expect(validation.tensorsVisited, 19);
    expect(validation.missingLayers, isEmpty);
    expect(validation.layers.every((layer) => layer.mapped), isTrue);

    expect(tokenEmbedding.ok, isTrue);
    expect(tokenEmbedding.tensorName, 'token_embd.weight');
    expect(tokenEmbedding.tensorType, 'F32');
    expect(tokenEmbedding.tokenId, 1);
    expect(tokenEmbedding.shape, [3, 2]);
    expect(tokenEmbedding.values, [4.0, 5.0, 6.0]);
    expect(tokenEmbedding.min, 4.0);
    expect(tokenEmbedding.max, 6.0);
    expect(tokenEmbedding.mean, 5.0);
    expect(tokenEmbedding.checksum, 32.0);

    final invRms =
        1.0 / math.sqrt(((4 * 4) + (5 * 5) + (6 * 6)) / 3 + 0.000001);
    final expectedRmsNorm = [4 * invRms, 10 * invRms, 18 * invRms];
    expect(rmsNorm.ok, isTrue);
    expect(rmsNorm.embeddingTensorName, 'token_embd.weight');
    expect(rmsNorm.normTensorName, 'blk.0.attn_norm.weight');
    expect(rmsNorm.embeddingLength, 3);
    expect(rmsNorm.meanSquare, closeTo(25.6666667, 0.000001));
    expect(rmsNorm.invRms, closeTo(invRms, 0.000001));
    expect(rmsNorm.values, hasLength(3));
    for (var i = 0; i < expectedRmsNorm.length; i++) {
      expect(rmsNorm.values[i], closeTo(expectedRmsNorm[i], 0.00001));
    }

    final expectedMatVec = [
      expectedRmsNorm[0] - expectedRmsNorm[2],
      expectedRmsNorm[0] * 0.5 + expectedRmsNorm[1] + expectedRmsNorm[2] * 1.5,
    ];
    expect(matVec.ok, isTrue);
    expect(matVec.weightTensorName, 'blk.0.attn_q.weight');
    expect(matVec.weightTensorType, 'F32');
    expect(matVec.inputLength, 3);
    expect(matVec.outputLength, 2);
    expect(matVec.weightShape, [3, 2]);
    expect(matVec.values, hasLength(2));
    for (var i = 0; i < expectedMatVec.length; i++) {
      expect(matVec.values[i], closeTo(expectedMatVec[i], 0.00001));
    }

    final cosine = math.cos(1);
    final sine = math.sin(1);
    final expectedRope = [
      expectedMatVec[0] * cosine - expectedMatVec[1] * sine,
      expectedMatVec[0] * sine + expectedMatVec[1] * cosine,
    ];
    expect(rope.ok, isTrue);
    expect(rope.weightTensorName, 'blk.0.attn_q.weight');
    expect(rope.outputLength, 2);
    expect(rope.position, 1);
    expect(rope.headDim, 2);
    expect(rope.headCount, 1);
    expect(rope.rotatedPairCount, 1);
    expect(rope.ropeTheta, 10000);
    expect(rope.weightShape, [3, 2]);
    expect(rope.values, hasLength(2));
    for (var i = 0; i < expectedRope.length; i++) {
      expect(rope.values[i], closeTo(expectedRope[i], 0.00001));
    }

    final expectedKeyRaw = [
      expectedRmsNorm[0] * 0.25 - expectedRmsNorm[1] * 0.5 + expectedRmsNorm[2],
      expectedRmsNorm[0] + expectedRmsNorm[2] * 0.5,
    ];
    final expectedKey = [
      expectedKeyRaw[0] * cosine - expectedKeyRaw[1] * sine,
      expectedKeyRaw[0] * sine + expectedKeyRaw[1] * cosine,
    ];
    final expectedValue = [
      -expectedRmsNorm[0] +
          expectedRmsNorm[1] * 0.25 +
          expectedRmsNorm[2] * 0.5,
      -expectedRmsNorm[1] * 0.75 + expectedRmsNorm[2] * 1.25,
    ];
    expect(kvCache.ok, isTrue);
    expect(kvCache.tokenIds, [0, 1]);
    expect(kvCache.sequenceLength, 2);
    expect(kvCache.keyWeightTensorName, 'blk.0.attn_k.weight');
    expect(kvCache.valueWeightTensorName, 'blk.0.attn_v.weight');
    expect(kvCache.keyWidth, 2);
    expect(kvCache.valueWidth, 2);
    expect(kvCache.headDim, 2);
    expect(kvCache.kvHeadCount, 1);
    expect(kvCache.cacheElementCount, 8);
    expect(kvCache.cacheBytesFp32, 32);
    expect(kvCache.startPosition, 0);
    expect(kvCache.readPosition, 1);
    expect(kvCache.readAbsolutePosition, 1);
    expect(kvCache.keyWeightShape, [3, 2]);
    expect(kvCache.valueWeightShape, [3, 2]);
    expect(kvCache.keyValues, hasLength(2));
    expect(kvCache.valueValues, hasLength(2));
    for (var i = 0; i < expectedKey.length; i++) {
      expect(kvCache.keyValues[i], closeTo(expectedKey[i], 0.00001));
      expect(kvCache.valueValues[i], closeTo(expectedValue[i], 0.00001));
    }

    final invRms0 =
        1.0 / math.sqrt(((1 * 1) + (2 * 2) + (3 * 3)) / 3 + 0.000001);
    final expectedRmsNorm0 = [1 * invRms0, 4 * invRms0, 9 * invRms0];
    final expectedKey0 = [
      expectedRmsNorm0[0] * 0.25 -
          expectedRmsNorm0[1] * 0.5 +
          expectedRmsNorm0[2],
      expectedRmsNorm0[0] + expectedRmsNorm0[2] * 0.5,
    ];
    final expectedValue0 = [
      -expectedRmsNorm0[0] +
          expectedRmsNorm0[1] * 0.25 +
          expectedRmsNorm0[2] * 0.5,
      -expectedRmsNorm0[1] * 0.75 + expectedRmsNorm0[2] * 1.25,
    ];
    final scale = 1 / math.sqrt(2);
    final expectedScores = [
      (expectedRope[0] * expectedKey0[0] + expectedRope[1] * expectedKey0[1]) *
          scale,
      (expectedRope[0] * expectedKey[0] + expectedRope[1] * expectedKey[1]) *
          scale,
    ];
    final maxScore = math.max(expectedScores[0], expectedScores[1]);
    final expScores = [
      math.exp(expectedScores[0] - maxScore),
      math.exp(expectedScores[1] - maxScore),
    ];
    final expSum = expScores[0] + expScores[1];
    final expectedProbabilities = [
      expScores[0] / expSum,
      expScores[1] / expSum,
    ];
    final expectedAttentionOutput = [
      expectedProbabilities[0] * expectedValue0[0] +
          expectedProbabilities[1] * expectedValue[0],
      expectedProbabilities[0] * expectedValue0[1] +
          expectedProbabilities[1] * expectedValue[1],
    ];
    final expectedMultiHeadRaw = [
      expectedMatVec[0],
      expectedMatVec[1],
      expectedRmsNorm[0] * -0.25 +
          expectedRmsNorm[1] * 0.75 +
          expectedRmsNorm[2] * 1.25,
      expectedRmsNorm[1] - expectedRmsNorm[2] * 0.5,
    ];
    final expectedMultiHeadQuery = [
      expectedRope[0],
      expectedRope[1],
      expectedMultiHeadRaw[2] * cosine - expectedMultiHeadRaw[3] * sine,
      expectedMultiHeadRaw[2] * sine + expectedMultiHeadRaw[3] * cosine,
    ];
    final expectedSecondHeadScores = [
      (expectedMultiHeadQuery[2] * expectedKey0[0] +
              expectedMultiHeadQuery[3] * expectedKey0[1]) *
          scale,
      (expectedMultiHeadQuery[2] * expectedKey[0] +
              expectedMultiHeadQuery[3] * expectedKey[1]) *
          scale,
    ];
    final secondHeadMaxScore = math.max(
      expectedSecondHeadScores[0],
      expectedSecondHeadScores[1],
    );
    final secondHeadExpScores = [
      math.exp(expectedSecondHeadScores[0] - secondHeadMaxScore),
      math.exp(expectedSecondHeadScores[1] - secondHeadMaxScore),
    ];
    final secondHeadExpSum = secondHeadExpScores[0] + secondHeadExpScores[1];
    final expectedSecondHeadProbabilities = [
      secondHeadExpScores[0] / secondHeadExpSum,
      secondHeadExpScores[1] / secondHeadExpSum,
    ];
    final expectedMultiHeadOutput = [
      expectedAttentionOutput[0],
      expectedAttentionOutput[1],
      expectedSecondHeadProbabilities[0] * expectedValue0[0] +
          expectedSecondHeadProbabilities[1] * expectedValue[0],
      expectedSecondHeadProbabilities[0] * expectedValue0[1] +
          expectedSecondHeadProbabilities[1] * expectedValue[1],
    ];
    final expectedAttentionProjection = _projectRows([
      [0.5, -0.25, 0.75, 0.0],
      [1.0, 0.0, -0.5, 0.25],
      [-0.25, 0.5, 0.0, 1.0],
    ], expectedMultiHeadOutput);
    final expectedPostAttention = [
      4.0 + expectedAttentionProjection[0],
      5.0 + expectedAttentionProjection[1],
      6.0 + expectedAttentionProjection[2],
    ];
    final expectedFfnNorm = _rmsNorm(expectedPostAttention, [
      2.0,
      3.0,
      4.0,
    ], 0.000001);
    final expectedGate = _projectRows([
      [0.25, 0.0, -0.5],
      [0.5, 0.25, 0.0],
      [-0.75, 0.5, 0.25],
      [0.0, -0.25, 0.75],
    ], expectedFfnNorm);
    final expectedUp = _projectRows([
      [1.0, -0.5, 0.25],
      [-0.25, 0.75, 0.5],
      [0.5, 0.0, -0.25],
      [0.25, 0.5, 1.0],
    ], expectedFfnNorm);
    final expectedFfnHidden = [
      for (var i = 0; i < expectedGate.length; i++)
        _silu(expectedGate[i]) * expectedUp[i],
    ];
    final expectedFfnOutput = _projectRows([
      [0.5, -0.25, 0.0, 0.75],
      [-0.5, 0.25, 1.0, 0.0],
      [0.25, 0.5, -0.75, 0.5],
    ], expectedFfnHidden);
    final expectedLayerOutput = [
      for (var i = 0; i < expectedPostAttention.length; i++)
        expectedPostAttention[i] + expectedFfnOutput[i],
    ];
    final qMultiRows = [
      [1.0, 0.0, -1.0],
      [0.5, 1.0, 1.5],
      [-0.25, 0.75, 1.25],
      [0.0, 1.0, -0.5],
    ];
    final keyRows = [
      [0.25, -0.5, 1.0],
      [1.0, 0.0, 0.5],
    ];
    final valueRows = [
      [-1.0, 0.25, 0.5],
      [0.0, -0.75, 1.25],
    ];
    final outputRows = [
      [0.5, -0.25, 0.75, 0.0],
      [1.0, 0.0, -0.5, 0.25],
      [-0.25, 0.5, 0.0, 1.0],
    ];
    final gateRows = [
      [0.25, 0.0, -0.5],
      [0.5, 0.25, 0.0],
      [-0.75, 0.5, 0.25],
      [0.0, -0.25, 0.75],
    ];
    final upRows = [
      [1.0, -0.5, 0.25],
      [-0.25, 0.75, 0.5],
      [0.5, 0.0, -0.25],
      [0.25, 0.5, 1.0],
    ];
    final downRows = [
      [0.5, -0.25, 0.0, 0.75],
      [-0.5, 0.25, 1.0, 0.0],
      [0.25, 0.5, -0.75, 0.5],
    ];
    final expectedStackLayer0 = _runExpectedLayer(
      inputStates: [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ],
      attnNormWeights: [1.0, 2.0, 3.0],
      ffnNormWeights: [2.0, 3.0, 4.0],
      qRows: qMultiRows,
      keyRows: keyRows,
      valueRows: valueRows,
      outputRows: outputRows,
      gateRows: gateRows,
      upRows: upRows,
      downRows: downRows,
      readIndex: 1,
    );
    final expectedStackLayer1 = _runExpectedLayer(
      inputStates: expectedStackLayer0.hiddenStates,
      attnNormWeights: [3.0, 4.0, 5.0],
      ffnNormWeights: [4.0, 5.0, 6.0],
      qRows: qMultiRows,
      keyRows: keyRows,
      valueRows: valueRows,
      outputRows: outputRows,
      gateRows: gateRows,
      upRows: upRows,
      downRows: downRows,
      readIndex: 1,
    );
    final expectedFinalNorm = _rmsNorm(expectedStackLayer1.outputValues, [
      1.5,
      0.5,
      2.0,
    ], 0.000001);
    final lmHeadRows = [
      [0.5, -0.25, 0.75],
      [-1.0, 0.5, 0.25],
    ];
    final expectedLogits = _projectRows(lmHeadRows, expectedFinalNorm);
    final expectedTopTokenIds =
        List<int>.generate(expectedLogits.length, (index) => index)..sort((
          left,
          right,
        ) {
          final byLogit = expectedLogits[right].compareTo(expectedLogits[left]);
          if (byLogit != 0) return byLogit;
          return left.compareTo(right);
        });
    final expectedReturnedTopTokenIds = expectedTopTokenIds.take(2).toList();
    final expectedGeneration = _runExpectedGeneration(
      promptTokenIds: [0, 1],
      maxNewTokens: 3,
      finalNormWeights: [1.5, 0.5, 2.0],
      lmHeadRows: lmHeadRows,
      qRows: qMultiRows,
      keyRows: keyRows,
      valueRows: valueRows,
      outputRows: outputRows,
      gateRows: gateRows,
      upRows: upRows,
      downRows: downRows,
    );
    expect(attention.ok, isTrue);
    expect(attention.tokenIds, [0, 1]);
    expect(attention.queryTokenId, 1);
    expect(attention.queryWeightTensorName, 'blk.0.attn_q.weight');
    expect(attention.keyWeightTensorName, 'blk.0.attn_k.weight');
    expect(attention.valueWeightTensorName, 'blk.0.attn_v.weight');
    expect(attention.sequenceLength, 2);
    expect(attention.attentionLength, 2);
    expect(attention.queryWidth, 2);
    expect(attention.keyWidth, 2);
    expect(attention.valueWidth, 2);
    expect(attention.headDim, 2);
    expect(attention.headIndex, 0);
    expect(attention.queryHeadCount, 1);
    expect(attention.kvHeadCount, 1);
    expect(attention.queryValues, hasLength(2));
    expect(attention.attentionScores, hasLength(2));
    expect(attention.attentionProbabilities, hasLength(2));
    expect(attention.outputValues, hasLength(2));
    for (var i = 0; i < 2; i++) {
      expect(attention.queryValues[i], closeTo(expectedRope[i], 0.00001));
      expect(attention.attentionScores[i], closeTo(expectedScores[i], 0.00001));
      expect(
        attention.attentionProbabilities[i],
        closeTo(expectedProbabilities[i], 0.00001),
      );
      expect(
        attention.outputValues[i],
        closeTo(expectedAttentionOutput[i], 0.00001),
      );
    }

    expect(multiHeadAttention.ok, isTrue);
    expect(multiHeadAttention.tokenIds, [0, 1]);
    expect(multiHeadAttention.queryTokenId, 1);
    expect(
      multiHeadAttention.queryWeightTensorName,
      'blk.0.attn_q_multi.weight',
    );
    expect(multiHeadAttention.keyWeightTensorName, 'blk.0.attn_k.weight');
    expect(multiHeadAttention.valueWeightTensorName, 'blk.0.attn_v.weight');
    expect(multiHeadAttention.sequenceLength, 2);
    expect(multiHeadAttention.attentionLength, 2);
    expect(multiHeadAttention.queryWidth, 4);
    expect(multiHeadAttention.keyWidth, 2);
    expect(multiHeadAttention.valueWidth, 2);
    expect(multiHeadAttention.headDim, 2);
    expect(multiHeadAttention.queryHeadCount, 2);
    expect(multiHeadAttention.kvHeadCount, 1);
    expect(multiHeadAttention.groupSize, 2);
    expect(multiHeadAttention.qHeadToKvHead, [0, 0]);
    expect(multiHeadAttention.queryValues, hasLength(4));
    expect(multiHeadAttention.firstHeadScores, hasLength(2));
    expect(multiHeadAttention.firstHeadProbabilities, hasLength(2));
    expect(multiHeadAttention.outputValues, hasLength(4));
    for (var i = 0; i < 4; i++) {
      expect(
        multiHeadAttention.queryValues[i],
        closeTo(expectedMultiHeadQuery[i], 0.00001),
      );
      expect(
        multiHeadAttention.outputValues[i],
        closeTo(expectedMultiHeadOutput[i], 0.00001),
      );
    }
    for (var i = 0; i < 2; i++) {
      expect(
        multiHeadAttention.firstHeadScores[i],
        closeTo(expectedScores[i], 0.00001),
      );
      expect(
        multiHeadAttention.firstHeadProbabilities[i],
        closeTo(expectedProbabilities[i], 0.00001),
      );
    }

    expect(layer.ok, isTrue);
    expect(layer.tokenIds, [0, 1]);
    expect(layer.queryTokenId, 1);
    expect(layer.attnNormTensorName, 'blk.0.attn_norm.weight');
    expect(layer.queryWeightTensorName, 'blk.0.attn_q_multi.weight');
    expect(layer.keyWeightTensorName, 'blk.0.attn_k.weight');
    expect(layer.valueWeightTensorName, 'blk.0.attn_v.weight');
    expect(layer.outputWeightTensorName, 'blk.0.attn_o.weight');
    expect(layer.ffnNormTensorName, 'blk.0.ffn_norm.weight');
    expect(layer.gateWeightTensorName, 'blk.0.ffn_gate.weight');
    expect(layer.upWeightTensorName, 'blk.0.ffn_up.weight');
    expect(layer.downWeightTensorName, 'blk.0.ffn_down.weight');
    expect(layer.sequenceLength, 2);
    expect(layer.attentionLength, 2);
    expect(layer.embeddingLength, 3);
    expect(layer.queryWidth, 4);
    expect(layer.keyWidth, 2);
    expect(layer.valueWidth, 2);
    expect(layer.attentionOutputWidth, 4);
    expect(layer.hiddenSize, 3);
    expect(layer.ffnWidth, 4);
    expect(layer.headDim, 2);
    expect(layer.queryHeadCount, 2);
    expect(layer.kvHeadCount, 1);
    expect(layer.groupSize, 2);
    expect(layer.qHeadToKvHead, [0, 0]);
    expect(layer.queryValues, hasLength(4));
    expect(layer.attentionValues, hasLength(4));
    expect(layer.attentionProjectedValues, hasLength(3));
    expect(layer.postAttentionValues, hasLength(3));
    expect(layer.gateValues, hasLength(4));
    expect(layer.upValues, hasLength(4));
    expect(layer.ffnHiddenValues, hasLength(4));
    expect(layer.ffnOutputValues, hasLength(3));
    expect(layer.outputValues, hasLength(3));
    for (var i = 0; i < 4; i++) {
      expect(layer.queryValues[i], closeTo(expectedMultiHeadQuery[i], 0.00001));
      expect(
        layer.attentionValues[i],
        closeTo(expectedMultiHeadOutput[i], 0.00001),
      );
      expect(layer.gateValues[i], closeTo(expectedGate[i], 0.00001));
      expect(layer.upValues[i], closeTo(expectedUp[i], 0.00001));
      expect(layer.ffnHiddenValues[i], closeTo(expectedFfnHidden[i], 0.00001));
    }
    for (var i = 0; i < 3; i++) {
      expect(
        layer.attentionProjectedValues[i],
        closeTo(expectedAttentionProjection[i], 0.00001),
      );
      expect(
        layer.postAttentionValues[i],
        closeTo(expectedPostAttention[i], 0.00001),
      );
      expect(layer.ffnOutputValues[i], closeTo(expectedFfnOutput[i], 0.00001));
      expect(layer.outputValues[i], closeTo(expectedLayerOutput[i], 0.00001));
    }

    expect(stack.ok, isTrue);
    expect(stack.tokenIds, [0, 1]);
    expect(stack.blockCount, 2);
    expect(stack.layerCount, 2);
    expect(stack.layersVisited, 2);
    expect(stack.tensorsVisited, 18);
    expect(stack.sequenceLength, 2);
    expect(stack.readPosition, 1);
    expect(stack.selectedTokenId, 1);
    expect(stack.hiddenSize, 3);
    expect(stack.queryWidth, 4);
    expect(stack.keyWidth, 2);
    expect(stack.valueWidth, 2);
    expect(stack.ffnWidth, 4);
    expect(stack.headDim, 2);
    expect(stack.queryHeadCount, 2);
    expect(stack.kvHeadCount, 1);
    expect(stack.groupSize, 2);
    expect(stack.kvCacheLayerCount, 2);
    expect(stack.kvCacheElementCount, 16);
    expect(stack.kvCacheBytesFp32, 64);
    expect(stack.qHeadToKvHead, [0, 0]);
    expect(stack.layerOutputChecksums, hasLength(2));
    expect(
      stack.layerOutputChecksums[0],
      closeTo(_checksum(expectedStackLayer0.outputValues), 0.00005),
    );
    expect(
      stack.layerOutputChecksums[1],
      closeTo(_checksum(expectedStackLayer1.outputValues), 0.00005),
    );
    expect(stack.lastLayerQueryValues, hasLength(4));
    expect(stack.lastLayerAttentionValues, hasLength(4));
    expect(stack.lastLayerAttentionProjectedValues, hasLength(3));
    expect(stack.lastLayerPostAttentionValues, hasLength(3));
    expect(stack.lastLayerGateValues, hasLength(4));
    expect(stack.lastLayerUpValues, hasLength(4));
    expect(stack.lastLayerFfnHiddenValues, hasLength(4));
    expect(stack.lastLayerFfnOutputValues, hasLength(3));
    expect(stack.outputValues, hasLength(3));
    for (var i = 0; i < 4; i++) {
      expect(
        stack.lastLayerQueryValues[i],
        closeTo(expectedStackLayer1.queryValues[i], 0.00001),
      );
      expect(
        stack.lastLayerAttentionValues[i],
        closeTo(expectedStackLayer1.attentionValues[i], 0.00001),
      );
      expect(
        stack.lastLayerGateValues[i],
        closeTo(expectedStackLayer1.gateValues[i], 0.00001),
      );
      expect(
        stack.lastLayerUpValues[i],
        closeTo(expectedStackLayer1.upValues[i], 0.00001),
      );
      expect(
        stack.lastLayerFfnHiddenValues[i],
        closeTo(expectedStackLayer1.ffnHiddenValues[i], 0.00001),
      );
    }
    for (var i = 0; i < 3; i++) {
      expect(
        stack.lastLayerAttentionProjectedValues[i],
        closeTo(expectedStackLayer1.attentionProjectedValues[i], 0.00001),
      );
      expect(
        stack.lastLayerPostAttentionValues[i],
        closeTo(expectedStackLayer1.postAttentionValues[i], 0.00001),
      );
      expect(
        stack.lastLayerFfnOutputValues[i],
        closeTo(expectedStackLayer1.ffnOutputValues[i], 0.00001),
      );
      expect(
        stack.outputValues[i],
        closeTo(expectedStackLayer1.outputValues[i], 0.00001),
      );
    }

    expect(nextToken.ok, isTrue);
    expect(nextToken.tokenIds, [0, 1]);
    expect(nextToken.blockCount, 2);
    expect(nextToken.layerCount, 2);
    expect(nextToken.layersVisited, 2);
    expect(nextToken.tensorsVisited, 20);
    expect(nextToken.sequenceLength, 2);
    expect(nextToken.readPosition, 1);
    expect(nextToken.selectedTokenId, 1);
    expect(nextToken.finalNormTensorName, 'output_norm.weight');
    expect(nextToken.lmHeadTensorName, 'output.weight');
    expect(nextToken.hiddenSize, 3);
    expect(nextToken.vocabSize, 2);
    expect(nextToken.lmHeadInputLength, 3);
    expect(nextToken.logitCount, 2);
    expect(nextToken.returnedTopK, 2);
    expect(nextToken.returnedHiddenValueCount, 3);
    expect(nextToken.returnedNormalizedValueCount, 3);
    expect(nextToken.returnedLogitValueCount, 2);
    expect(nextToken.nextTokenId, expectedReturnedTopTokenIds.first);
    expect(nextToken.topTokenIds, expectedReturnedTopTokenIds);
    for (var i = 0; i < 3; i++) {
      expect(
        nextToken.hiddenValues[i],
        closeTo(expectedStackLayer1.outputValues[i], 0.00001),
      );
      expect(
        nextToken.normalizedValues[i],
        closeTo(expectedFinalNorm[i], 0.00001),
      );
    }
    for (var i = 0; i < expectedLogits.length; i++) {
      expect(nextToken.logitValues[i], closeTo(expectedLogits[i], 0.00001));
    }
    for (var i = 0; i < expectedReturnedTopTokenIds.length; i++) {
      final tokenId = expectedReturnedTopTokenIds[i];
      expect(nextToken.topLogits[i], closeTo(expectedLogits[tokenId], 0.00001));
    }

    expect(generated.ok, isTrue);
    expect(generated.cacheReused, isTrue);
    expect(generated.tokenIds, [0, 1]);
    expect(generated.generatedTokenCount, 3);
    expect(generated.totalTokenCount, 5);
    expect(generated.maxNewTokens, 3);
    expect(generated.generatedTokenIds, expectedGeneration.generatedTokenIds);
    expect(generated.allTokenIds, expectedGeneration.allTokenIds);
    expect(generated.layersVisited, 6);
    expect(generated.tensorsVisited, 60);
    expect(generated.initialKvCacheTokenCount, 2);
    expect(generated.finalKvCacheTokenCount, 4);
    expect(generated.initialKvCacheElementCount, 16);
    expect(generated.finalKvCacheElementCount, 32);
    expect(generated.initialKvCacheBytesFp32, 64);
    expect(generated.finalKvCacheBytesFp32, 128);
    expect(generated.logitCount, 2);
    expect(generated.returnedTopK, 2);
    for (var i = 0; i < generated.generatedTokenLogits.length; i++) {
      expect(
        generated.generatedTokenLogits[i],
        closeTo(expectedGeneration.generatedTokenLogits[i], 0.00001),
      );
    }
    for (var i = 0; i < expectedGeneration.lastHidden.length; i++) {
      expect(
        generated.hiddenValues[i],
        closeTo(expectedGeneration.lastHidden[i], 0.00001),
      );
      expect(
        generated.normalizedValues[i],
        closeTo(expectedGeneration.lastNormalized[i], 0.00001),
      );
    }
    for (var i = 0; i < expectedGeneration.lastLogits.length; i++) {
      expect(
        generated.logitValues[i],
        closeTo(expectedGeneration.lastLogits[i], 0.00001),
      );
    }

    expect(session.ok, isTrue);
    expect(session.prefilled, isTrue);
    expect(session.sessionId, greaterThan(0));
    expect(session.tokenIds, [0, 1]);
    expect(session.allTokenIds, [0, 1]);
    expect(session.promptTokenCount, 2);
    expect(session.totalTokenCount, 2);
    expect(session.generatedTokenCount, 0);
    expect(session.nextPosition, 2);
    expect(session.kvCacheTokenCount, 2);
    expect(session.kvCacheElementCount, 16);
    expect(session.returnedHiddenValueCount, 3);
    for (var i = 0; i < expectedStackLayer1.outputValues.length; i++) {
      expect(
        session.hiddenValues[i],
        closeTo(expectedStackLayer1.outputValues[i], 0.00001),
      );
    }

    expect(sessionDecodes, hasLength(3));
    for (var step = 0; step < sessionDecodes.length; step++) {
      final decoded = sessionDecodes[step];
      final stepExpected = _runExpectedGeneration(
        promptTokenIds: [0, 1],
        maxNewTokens: step + 1,
        finalNormWeights: [1.5, 0.5, 2.0],
        lmHeadRows: lmHeadRows,
        qRows: qMultiRows,
        keyRows: keyRows,
        valueRows: valueRows,
        outputRows: outputRows,
        gateRows: gateRows,
        upRows: upRows,
        downRows: downRows,
      );
      final sortedTokenIds =
          List<int>.generate(stepExpected.lastLogits.length, (index) => index)
            ..sort((left, right) {
              final byLogit = stepExpected.lastLogits[right].compareTo(
                stepExpected.lastLogits[left],
              );
              if (byLogit != 0) return byLogit;
              return left.compareTo(right);
            });

      expect(decoded.ok, isTrue);
      expect(decoded.sessionId, session.sessionId);
      expect(decoded.decodedPosition, 2 + step);
      expect(decoded.nextPosition, 3 + step);
      expect(decoded.generatedTokenId, stepExpected.generatedTokenIds.last);
      expect(decoded.generatedTokenCount, step + 1);
      expect(decoded.totalTokenCount, 3 + step);
      expect(decoded.generatedTokenIds, stepExpected.generatedTokenIds);
      expect(decoded.allTokenIds, stepExpected.allTokenIds);
      expect(decoded.kvCacheTokenCount, 3 + step);
      expect(decoded.kvCacheElementCount, 16 + ((step + 1) * 8));
      expect(decoded.returnedTopK, 2);
      expect(decoded.topTokenIds, sortedTokenIds.take(2).toList());
      for (var i = 0; i < stepExpected.generatedTokenLogits.length; i++) {
        expect(
          decoded.generatedTokenLogits[i],
          closeTo(stepExpected.generatedTokenLogits[i], 0.00001),
        );
      }
      for (var i = 0; i < stepExpected.lastHidden.length; i++) {
        expect(
          decoded.hiddenValues[i],
          closeTo(stepExpected.lastHidden[i], 0.00001),
        );
        expect(
          decoded.normalizedValues[i],
          closeTo(stepExpected.lastNormalized[i], 0.00001),
        );
      }
      for (var i = 0; i < stepExpected.lastLogits.length; i++) {
        expect(
          decoded.logitValues[i],
          closeTo(stepExpected.lastLogits[i], 0.00001),
        );
      }
    }

    expect(closedSession.ok, isTrue);
    expect(closedSession.sessionId, session.sessionId);
    expect(closedSession.closed, isTrue);
  });
}

List<int> _tinyGgufFixture() {
  final writer = _ByteWriter();
  writer.uint32(0x46554747); // GGUF
  writer.uint32(3); // version
  writer.uint64(22); // tensor count
  writer.uint64(3); // metadata count

  writer.metadataString('general.architecture', 'qwen2');
  writer.metadataUint32('general.alignment', 32);
  writer.metadataUint32('qwen2.block_count', 2);

  writer.string('token_embd.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // embedding length
  writer.uint64(2); // vocab size
  writer.uint32(0); // F32
  writer.uint64(0); // relative tensor data offset

  final tensorNames = [
    'blk.0.attn_norm.weight',
    'blk.0.ffn_norm.weight',
    'blk.1.attn_norm.weight',
    'blk.1.ffn_norm.weight',
  ];
  for (var i = 0; i < tensorNames.length; i++) {
    writer.string(tensorNames[i]);
    writer.uint32(1); // dimensions
    writer.uint64(3); // shape [3]
    writer.uint32(0); // F32
    writer.uint64((i + 1) * 32); // relative tensor data offset
  }

  writer.string('blk.0.attn_q.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(2); // output rows
  writer.uint32(0); // F32
  writer.uint64(160); // relative tensor data offset

  writer.string('blk.0.attn_k.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(2); // output rows
  writer.uint32(0); // F32
  writer.uint64(192); // relative tensor data offset

  writer.string('blk.0.attn_v.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(2); // output rows
  writer.uint32(0); // F32
  writer.uint64(224); // relative tensor data offset

  writer.string('blk.0.attn_q_multi.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(4); // output rows
  writer.uint32(0); // F32
  writer.uint64(256); // relative tensor data offset

  writer.string('blk.0.attn_o.weight');
  writer.uint32(2); // dimensions
  writer.uint64(4); // attention width
  writer.uint64(3); // output rows
  writer.uint32(0); // F32
  writer.uint64(320); // relative tensor data offset

  writer.string('blk.0.ffn_gate.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(4); // FFN rows
  writer.uint32(0); // F32
  writer.uint64(384); // relative tensor data offset

  writer.string('blk.0.ffn_up.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(4); // FFN rows
  writer.uint32(0); // F32
  writer.uint64(448); // relative tensor data offset

  writer.string('blk.0.ffn_down.weight');
  writer.uint32(2); // dimensions
  writer.uint64(4); // FFN width
  writer.uint64(3); // output rows
  writer.uint32(0); // F32
  writer.uint64(512); // relative tensor data offset

  writer.string('blk.1.attn_q_multi.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(4); // output rows
  writer.uint32(0); // F32
  writer.uint64(576); // relative tensor data offset

  writer.string('blk.1.attn_k.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(2); // output rows
  writer.uint32(0); // F32
  writer.uint64(640); // relative tensor data offset

  writer.string('blk.1.attn_v.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(2); // output rows
  writer.uint32(0); // F32
  writer.uint64(672); // relative tensor data offset

  writer.string('blk.1.attn_o.weight');
  writer.uint32(2); // dimensions
  writer.uint64(4); // attention width
  writer.uint64(3); // output rows
  writer.uint32(0); // F32
  writer.uint64(704); // relative tensor data offset

  writer.string('blk.1.ffn_gate.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(4); // FFN rows
  writer.uint32(0); // F32
  writer.uint64(768); // relative tensor data offset

  writer.string('blk.1.ffn_up.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // input width
  writer.uint64(4); // FFN rows
  writer.uint32(0); // F32
  writer.uint64(832); // relative tensor data offset

  writer.string('blk.1.ffn_down.weight');
  writer.uint32(2); // dimensions
  writer.uint64(4); // FFN width
  writer.uint64(3); // output rows
  writer.uint32(0); // F32
  writer.uint64(896); // relative tensor data offset

  writer.string('output_norm.weight');
  writer.uint32(1); // dimensions
  writer.uint64(3); // hidden size
  writer.uint32(0); // F32
  writer.uint64(960); // relative tensor data offset

  writer.string('output.weight');
  writer.uint32(2); // dimensions
  writer.uint64(3); // hidden size
  writer.uint64(2); // vocab/logit rows
  writer.uint32(0); // F32
  writer.uint64(992); // relative tensor data offset

  writer.align(32);
  for (final value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    writer.float32(value);
  }
  for (var i = 0; i < tensorNames.length; i++) {
    writer.align(32);
    for (final value in [1.0 + i, 2.0 + i, 3.0 + i]) {
      writer.float32(value);
    }
  }
  writer.align(32);
  for (final value in [1.0, 0.0, -1.0, 0.5, 1.0, 1.5]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [0.25, -0.5, 1.0, 1.0, 0.0, 0.5]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [-1.0, 0.25, 0.5, 0.0, -0.75, 1.25]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    1.0,
    0.0,
    -1.0,
    0.5,
    1.0,
    1.5,
    -0.25,
    0.75,
    1.25,
    0.0,
    1.0,
    -0.5,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    0.5,
    -0.25,
    0.75,
    0.0,
    1.0,
    0.0,
    -0.5,
    0.25,
    -0.25,
    0.5,
    0.0,
    1.0,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    0.25,
    0.0,
    -0.5,
    0.5,
    0.25,
    0.0,
    -0.75,
    0.5,
    0.25,
    0.0,
    -0.25,
    0.75,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    1.0,
    -0.5,
    0.25,
    -0.25,
    0.75,
    0.5,
    0.5,
    0.0,
    -0.25,
    0.25,
    0.5,
    1.0,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    0.5,
    -0.25,
    0.0,
    0.75,
    -0.5,
    0.25,
    1.0,
    0.0,
    0.25,
    0.5,
    -0.75,
    0.5,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    1.0,
    0.0,
    -1.0,
    0.5,
    1.0,
    1.5,
    -0.25,
    0.75,
    1.25,
    0.0,
    1.0,
    -0.5,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [0.25, -0.5, 1.0, 1.0, 0.0, 0.5]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [-1.0, 0.25, 0.5, 0.0, -0.75, 1.25]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    0.5,
    -0.25,
    0.75,
    0.0,
    1.0,
    0.0,
    -0.5,
    0.25,
    -0.25,
    0.5,
    0.0,
    1.0,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    0.25,
    0.0,
    -0.5,
    0.5,
    0.25,
    0.0,
    -0.75,
    0.5,
    0.25,
    0.0,
    -0.25,
    0.75,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    1.0,
    -0.5,
    0.25,
    -0.25,
    0.75,
    0.5,
    0.5,
    0.0,
    -0.25,
    0.25,
    0.5,
    1.0,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [
    0.5,
    -0.25,
    0.0,
    0.75,
    -0.5,
    0.25,
    1.0,
    0.0,
    0.25,
    0.5,
    -0.75,
    0.5,
  ]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [1.5, 0.5, 2.0]) {
    writer.float32(value);
  }
  writer.align(32);
  for (final value in [0.5, -0.25, 0.75, -1.0, 0.5, 0.25]) {
    writer.float32(value);
  }

  return writer.takeBytes();
}

List<double> _projectRows(List<List<double>> rows, List<double> input) {
  return [
    for (final row in rows)
      row.indexed.fold<double>(
        0,
        (sum, entry) => sum + entry.$2 * input[entry.$1],
      ),
  ];
}

List<double> _rmsNorm(
  List<double> input,
  List<double> weights,
  double epsilon,
) {
  final meanSquare =
      input.fold<double>(0, (sum, value) => sum + value * value) / input.length;
  final invRms = 1.0 / math.sqrt(meanSquare + epsilon);
  return [
    for (var i = 0; i < input.length; i++) input[i] * invRms * weights[i],
  ];
}

double _silu(double value) {
  return value / (1.0 + math.exp(-value));
}

class _ExpectedLayer {
  const _ExpectedLayer({
    required this.hiddenStates,
    required this.queryValues,
    required this.attentionValues,
    required this.attentionProjectedValues,
    required this.postAttentionValues,
    required this.gateValues,
    required this.upValues,
    required this.ffnHiddenValues,
    required this.ffnOutputValues,
    required this.outputValues,
  });

  final List<List<double>> hiddenStates;
  final List<double> queryValues;
  final List<double> attentionValues;
  final List<double> attentionProjectedValues;
  final List<double> postAttentionValues;
  final List<double> gateValues;
  final List<double> upValues;
  final List<double> ffnHiddenValues;
  final List<double> ffnOutputValues;
  final List<double> outputValues;
}

class _ExpectedGeneration {
  const _ExpectedGeneration({
    required this.generatedTokenIds,
    required this.allTokenIds,
    required this.generatedTokenLogits,
    required this.lastHidden,
    required this.lastNormalized,
    required this.lastLogits,
  });

  final List<int> generatedTokenIds;
  final List<int> allTokenIds;
  final List<double> generatedTokenLogits;
  final List<double> lastHidden;
  final List<double> lastNormalized;
  final List<double> lastLogits;
}

_ExpectedGeneration _runExpectedGeneration({
  required List<int> promptTokenIds,
  required int maxNewTokens,
  required List<double> finalNormWeights,
  required List<List<double>> lmHeadRows,
  required List<List<double>> qRows,
  required List<List<double>> keyRows,
  required List<List<double>> valueRows,
  required List<List<double>> outputRows,
  required List<List<double>> gateRows,
  required List<List<double>> upRows,
  required List<List<double>> downRows,
}) {
  final allTokenIds = [...promptTokenIds];
  final generatedTokenIds = <int>[];
  final generatedTokenLogits = <double>[];
  var lastHidden = <double>[];
  var lastNormalized = <double>[];
  var lastLogits = <double>[];

  for (var step = 0; step < maxNewTokens; step++) {
    final inputStates = [
      for (final tokenId in allTokenIds) _embeddingForToken(tokenId),
    ];
    final layer0 = _runExpectedLayer(
      inputStates: inputStates,
      attnNormWeights: [1.0, 2.0, 3.0],
      ffnNormWeights: [2.0, 3.0, 4.0],
      qRows: qRows,
      keyRows: keyRows,
      valueRows: valueRows,
      outputRows: outputRows,
      gateRows: gateRows,
      upRows: upRows,
      downRows: downRows,
      readIndex: allTokenIds.length - 1,
    );
    final layer1 = _runExpectedLayer(
      inputStates: layer0.hiddenStates,
      attnNormWeights: [3.0, 4.0, 5.0],
      ffnNormWeights: [4.0, 5.0, 6.0],
      qRows: qRows,
      keyRows: keyRows,
      valueRows: valueRows,
      outputRows: outputRows,
      gateRows: gateRows,
      upRows: upRows,
      downRows: downRows,
      readIndex: allTokenIds.length - 1,
    );
    lastHidden = layer1.outputValues;
    lastNormalized = _rmsNorm(lastHidden, finalNormWeights, 0.000001);
    lastLogits = _projectRows(lmHeadRows, lastNormalized);

    var nextTokenId = 0;
    for (var i = 1; i < lastLogits.length; i++) {
      if (lastLogits[i] > lastLogits[nextTokenId]) {
        nextTokenId = i;
      }
    }
    generatedTokenIds.add(nextTokenId);
    generatedTokenLogits.add(lastLogits[nextTokenId]);
    allTokenIds.add(nextTokenId);
  }

  return _ExpectedGeneration(
    generatedTokenIds: generatedTokenIds,
    allTokenIds: allTokenIds,
    generatedTokenLogits: generatedTokenLogits,
    lastHidden: lastHidden,
    lastNormalized: lastNormalized,
    lastLogits: lastLogits,
  );
}

List<double> _embeddingForToken(int tokenId) {
  return switch (tokenId) {
    0 => [1.0, 2.0, 3.0],
    1 => [4.0, 5.0, 6.0],
    _ => throw StateError('Unexpected tiny fixture token id $tokenId'),
  };
}

_ExpectedLayer _runExpectedLayer({
  required List<List<double>> inputStates,
  required List<double> attnNormWeights,
  required List<double> ffnNormWeights,
  required List<List<double>> qRows,
  required List<List<double>> keyRows,
  required List<List<double>> valueRows,
  required List<List<double>> outputRows,
  required List<List<double>> gateRows,
  required List<List<double>> upRows,
  required List<List<double>> downRows,
  required int readIndex,
}) {
  final attnInputs = [
    for (final state in inputStates) _rmsNorm(state, attnNormWeights, 0.000001),
  ];
  final keyCache = <List<double>>[];
  final valueCache = <List<double>>[];
  for (var position = 0; position < inputStates.length; position++) {
    keyCache.add(
      _applyRope(_projectRows(keyRows, attnInputs[position]), position),
    );
    valueCache.add(_projectRows(valueRows, attnInputs[position]));
  }

  final nextStates = <List<double>>[];
  late List<double> selectedQuery;
  late List<double> selectedAttention;
  late List<double> selectedAttentionProjection;
  late List<double> selectedPostAttention;
  late List<double> selectedGate;
  late List<double> selectedUp;
  late List<double> selectedFfnHidden;
  late List<double> selectedFfnOutput;
  late List<double> selectedOutput;

  for (var position = 0; position < inputStates.length; position++) {
    final query = _applyRope(
      _projectRows(qRows, attnInputs[position]),
      position,
    );
    final attention = List<double>.filled(query.length, 0);
    for (var queryHead = 0; queryHead < query.length ~/ 2; queryHead++) {
      final queryOffset = queryHead * 2;
      final scores = <double>[];
      for (var tokenIndex = 0; tokenIndex <= position; tokenIndex++) {
        scores.add(
          (query[queryOffset] * keyCache[tokenIndex][0] +
                  query[queryOffset + 1] * keyCache[tokenIndex][1]) /
              math.sqrt(2),
        );
      }
      final probabilities = _softmax(scores);
      for (var tokenIndex = 0; tokenIndex <= position; tokenIndex++) {
        attention[queryOffset] +=
            probabilities[tokenIndex] * valueCache[tokenIndex][0];
        attention[queryOffset + 1] +=
            probabilities[tokenIndex] * valueCache[tokenIndex][1];
      }
    }

    final attentionProjection = _projectRows(outputRows, attention);
    final postAttention = [
      for (var i = 0; i < inputStates[position].length; i++)
        inputStates[position][i] + attentionProjection[i],
    ];
    final ffnNorm = _rmsNorm(postAttention, ffnNormWeights, 0.000001);
    final gate = _projectRows(gateRows, ffnNorm);
    final up = _projectRows(upRows, ffnNorm);
    final ffnHidden = [
      for (var i = 0; i < gate.length; i++) _silu(gate[i]) * up[i],
    ];
    final ffnOutput = _projectRows(downRows, ffnHidden);
    final output = [
      for (var i = 0; i < postAttention.length; i++)
        postAttention[i] + ffnOutput[i],
    ];
    nextStates.add(output);

    if (position == readIndex) {
      selectedQuery = query;
      selectedAttention = attention;
      selectedAttentionProjection = attentionProjection;
      selectedPostAttention = postAttention;
      selectedGate = gate;
      selectedUp = up;
      selectedFfnHidden = ffnHidden;
      selectedFfnOutput = ffnOutput;
      selectedOutput = output;
    }
  }

  return _ExpectedLayer(
    hiddenStates: nextStates,
    queryValues: selectedQuery,
    attentionValues: selectedAttention,
    attentionProjectedValues: selectedAttentionProjection,
    postAttentionValues: selectedPostAttention,
    gateValues: selectedGate,
    upValues: selectedUp,
    ffnHiddenValues: selectedFfnHidden,
    ffnOutputValues: selectedFfnOutput,
    outputValues: selectedOutput,
  );
}

List<double> _applyRope(List<double> values, int position) {
  final output = [...values];
  for (var i = 0; i < output.length; i += 2) {
    final x0 = output[i];
    final x1 = output[i + 1];
    final cosine = math.cos(position);
    final sine = math.sin(position);
    output[i] = x0 * cosine - x1 * sine;
    output[i + 1] = x0 * sine + x1 * cosine;
  }
  return output;
}

List<double> _softmax(List<double> scores) {
  final maxScore = scores.reduce(math.max);
  final expScores = [for (final score in scores) math.exp(score - maxScore)];
  final sum = expScores.reduce((left, right) => left + right);
  return [for (final value in expScores) value / sum];
}

double _checksum(List<double> values) {
  var checksum = 0.0;
  for (var i = 0; i < values.length; i++) {
    checksum += values[i] * (i + 1);
  }
  return checksum;
}

class _ByteWriter {
  final BytesBuilder _bytes = BytesBuilder(copy: false);

  int get length => _bytes.length;

  void uint32(int value) {
    final data = ByteData(4)..setUint32(0, value, Endian.little);
    _bytes.add(data.buffer.asUint8List());
  }

  void uint64(int value) {
    final data = ByteData(8)..setUint64(0, value, Endian.little);
    _bytes.add(data.buffer.asUint8List());
  }

  void float32(double value) {
    final data = ByteData(4)..setFloat32(0, value, Endian.little);
    _bytes.add(data.buffer.asUint8List());
  }

  void string(String value) {
    final encoded = Uint8List.fromList(value.codeUnits);
    uint64(encoded.length);
    _bytes.add(encoded);
  }

  void metadataString(String key, String value) {
    string(key);
    uint32(8);
    string(value);
  }

  void metadataUint32(String key, int value) {
    string(key);
    uint32(4);
    uint32(value);
  }

  void align(int alignment) {
    final remainder = length % alignment;
    if (remainder == 0) return;
    _bytes.add(Uint8List(alignment - remainder));
  }

  List<int> takeBytes() => _bytes.takeBytes();
}
