import 'dart:convert';

import 'package:flutter/services.dart';

class GgufRunnerException implements Exception {
  const GgufRunnerException(this.message);

  final String message;

  @override
  String toString() => message;
}

class GgufRunnerService {
  static const MethodChannel _deviceChannel = MethodChannel(
    'local_doc_qa/device',
  );

  Future<GgufInspection> inspectGguf(String path) async {
    final inspection = GgufInspection.fromJson(
      await _invokeJson('inspectGguf', path),
    );
    if (!inspection.ok) {
      throw GgufRunnerException(inspection.error ?? 'GGUF inspection failed');
    }
    return inspection;
  }

  Future<GgufLayerStreamingValidation> validateLayerStreaming(
    String path,
  ) async {
    final validation = GgufLayerStreamingValidation.fromJson(
      await _invokeJson('validateGgufLayerStreaming', path),
    );
    if (!validation.ok) {
      throw GgufRunnerException(
        validation.error ?? 'GGUF layer streaming validation failed',
      );
    }
    return validation;
  }

  Future<GgufTokenEmbedding> readTokenEmbedding({
    required String path,
    required int tokenId,
    int maxValues = 16,
  }) async {
    final embedding = GgufTokenEmbedding.fromJson(
      await _invokeJson('readGgufTokenEmbedding', path, {
        'tokenId': tokenId,
        'maxValues': maxValues,
      }),
    );
    if (!embedding.ok) {
      throw GgufRunnerException(
        embedding.error ?? 'GGUF token embedding read failed',
      );
    }
    return embedding;
  }

  Future<GgufRmsNormResult> rmsNormTokenEmbedding({
    required String path,
    required int tokenId,
    required String normTensorName,
    double epsilon = 0.000001,
    int maxValues = 16,
  }) async {
    final result = GgufRmsNormResult.fromJson(
      await _invokeJson('rmsNormGgufTokenEmbedding', path, {
        'tokenId': tokenId,
        'normTensorName': normTensorName,
        'epsilon': epsilon,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF RMSNorm failed');
    }
    return result;
  }

  Future<GgufMatVecResult> matVecRmsNormTokenEmbedding({
    required String path,
    required int tokenId,
    required String normTensorName,
    required String weightTensorName,
    double epsilon = 0.000001,
    int maxValues = 16,
  }) async {
    final result = GgufMatVecResult.fromJson(
      await _invokeJson('matVecGgufRmsNormTokenEmbedding', path, {
        'tokenId': tokenId,
        'normTensorName': normTensorName,
        'weightTensorName': weightTensorName,
        'epsilon': epsilon,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF MatVec failed');
    }
    return result;
  }

  Future<GgufRopeResult> ropeMatVecRmsNormTokenEmbedding({
    required String path,
    required int tokenId,
    required String normTensorName,
    required String weightTensorName,
    required int position,
    required int headDim,
    double epsilon = 0.000001,
    double ropeTheta = 0,
    int maxValues = 16,
  }) async {
    final result = GgufRopeResult.fromJson(
      await _invokeJson('ropeGgufMatVecRmsNormTokenEmbedding', path, {
        'tokenId': tokenId,
        'normTensorName': normTensorName,
        'weightTensorName': weightTensorName,
        'epsilon': epsilon,
        'position': position,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF RoPE MatVec failed');
    }
    return result;
  }

  Future<GgufKvCacheResult> kvCacheRmsNormTokenEmbeddings({
    required String path,
    required List<int> tokenIds,
    required String normTensorName,
    required String keyWeightTensorName,
    required String valueWeightTensorName,
    required int headDim,
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    double ropeTheta = 0,
    int maxValues = 16,
  }) async {
    final result = GgufKvCacheResult.fromJson(
      await _invokeJson('kvCacheGgufRmsNormTokenEmbeddings', path, {
        'tokenIds': tokenIds,
        'normTensorName': normTensorName,
        'keyWeightTensorName': keyWeightTensorName,
        'valueWeightTensorName': valueWeightTensorName,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'readPosition': readPosition,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF KV cache failed');
    }
    return result;
  }

  Future<GgufAttentionResult> attentionSingleHead({
    required String path,
    required List<int> tokenIds,
    required int queryTokenId,
    required String normTensorName,
    required String queryWeightTensorName,
    required String keyWeightTensorName,
    required String valueWeightTensorName,
    required int queryPosition,
    required int headDim,
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    int attentionLength = -1,
    int headIndex = 0,
    double ropeTheta = 0,
    int maxValues = 16,
  }) async {
    final result = GgufAttentionResult.fromJson(
      await _invokeJson('attentionGgufSingleHead', path, {
        'tokenIds': tokenIds,
        'queryTokenId': queryTokenId,
        'normTensorName': normTensorName,
        'queryWeightTensorName': queryWeightTensorName,
        'keyWeightTensorName': keyWeightTensorName,
        'valueWeightTensorName': valueWeightTensorName,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'queryPosition': queryPosition,
        'readPosition': readPosition,
        'attentionLength': attentionLength,
        'headDim': headDim,
        'headIndex': headIndex,
        'ropeTheta': ropeTheta,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF attention failed');
    }
    return result;
  }

  Future<GgufMultiHeadAttentionResult> attentionMultiHead({
    required String path,
    required List<int> tokenIds,
    required int queryTokenId,
    required String normTensorName,
    required String queryWeightTensorName,
    required String keyWeightTensorName,
    required String valueWeightTensorName,
    required int queryPosition,
    required int headDim,
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    int attentionLength = -1,
    double ropeTheta = 0,
    int maxValues = 16,
  }) async {
    final result = GgufMultiHeadAttentionResult.fromJson(
      await _invokeJson('attentionGgufMultiHead', path, {
        'tokenIds': tokenIds,
        'queryTokenId': queryTokenId,
        'normTensorName': normTensorName,
        'queryWeightTensorName': queryWeightTensorName,
        'keyWeightTensorName': keyWeightTensorName,
        'valueWeightTensorName': valueWeightTensorName,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'queryPosition': queryPosition,
        'readPosition': readPosition,
        'attentionLength': attentionLength,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(
        result.error ?? 'GGUF multi-head attention failed',
      );
    }
    return result;
  }

  Future<GgufTransformerLayerResult> transformerLayer({
    required String path,
    required List<int> tokenIds,
    required int queryTokenId,
    required String attnNormTensorName,
    required String queryWeightTensorName,
    required String keyWeightTensorName,
    required String valueWeightTensorName,
    required String outputWeightTensorName,
    required String ffnNormTensorName,
    required String gateWeightTensorName,
    required String upWeightTensorName,
    required String downWeightTensorName,
    required int queryPosition,
    required int headDim,
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    int attentionLength = -1,
    double ropeTheta = 0,
    int maxValues = 16,
  }) async {
    final result = GgufTransformerLayerResult.fromJson(
      await _invokeJson('transformerLayerGguf', path, {
        'tokenIds': tokenIds,
        'queryTokenId': queryTokenId,
        'attnNormTensorName': attnNormTensorName,
        'queryWeightTensorName': queryWeightTensorName,
        'keyWeightTensorName': keyWeightTensorName,
        'valueWeightTensorName': valueWeightTensorName,
        'outputWeightTensorName': outputWeightTensorName,
        'ffnNormTensorName': ffnNormTensorName,
        'gateWeightTensorName': gateWeightTensorName,
        'upWeightTensorName': upWeightTensorName,
        'downWeightTensorName': downWeightTensorName,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'queryPosition': queryPosition,
        'readPosition': readPosition,
        'attentionLength': attentionLength,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(
        result.error ?? 'GGUF transformer layer failed',
      );
    }
    return result;
  }

  Future<GgufTransformerStackResult> transformerStack({
    required String path,
    required List<int> tokenIds,
    required int headDim,
    String attnNormTensorSuffix = 'attn_norm.weight',
    String queryWeightTensorSuffix = 'attn_q.weight',
    String keyWeightTensorSuffix = 'attn_k.weight',
    String valueWeightTensorSuffix = 'attn_v.weight',
    String outputWeightTensorSuffix = 'attn_o.weight',
    String ffnNormTensorSuffix = 'ffn_norm.weight',
    String gateWeightTensorSuffix = 'ffn_gate.weight',
    String upWeightTensorSuffix = 'ffn_up.weight',
    String downWeightTensorSuffix = 'ffn_down.weight',
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    int layerCount = -1,
    double ropeTheta = 0,
    int maxValues = 16,
  }) async {
    final result = GgufTransformerStackResult.fromJson(
      await _invokeJson('transformerStackGguf', path, {
        'tokenIds': tokenIds,
        'attnNormTensorSuffix': attnNormTensorSuffix,
        'queryWeightTensorSuffix': queryWeightTensorSuffix,
        'keyWeightTensorSuffix': keyWeightTensorSuffix,
        'valueWeightTensorSuffix': valueWeightTensorSuffix,
        'outputWeightTensorSuffix': outputWeightTensorSuffix,
        'ffnNormTensorSuffix': ffnNormTensorSuffix,
        'gateWeightTensorSuffix': gateWeightTensorSuffix,
        'upWeightTensorSuffix': upWeightTensorSuffix,
        'downWeightTensorSuffix': downWeightTensorSuffix,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'readPosition': readPosition,
        'layerCount': layerCount,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(
        result.error ?? 'GGUF transformer stack failed',
      );
    }
    return result;
  }

  Future<GgufNextTokenResult> generateNextToken({
    required String path,
    required List<int> tokenIds,
    required int headDim,
    String attnNormTensorSuffix = 'attn_norm.weight',
    String queryWeightTensorSuffix = 'attn_q.weight',
    String keyWeightTensorSuffix = 'attn_k.weight',
    String valueWeightTensorSuffix = 'attn_v.weight',
    String outputWeightTensorSuffix = 'attn_o.weight',
    String ffnNormTensorSuffix = 'ffn_norm.weight',
    String gateWeightTensorSuffix = 'ffn_gate.weight',
    String upWeightTensorSuffix = 'ffn_up.weight',
    String downWeightTensorSuffix = 'ffn_down.weight',
    String finalNormTensorName = 'output_norm.weight',
    String lmHeadTensorName = 'output.weight',
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    int layerCount = -1,
    double ropeTheta = 0,
    int topK = 5,
    int maxValues = 16,
  }) async {
    final result = GgufNextTokenResult.fromJson(
      await _invokeJson('generateNextTokenGguf', path, {
        'tokenIds': tokenIds,
        'attnNormTensorSuffix': attnNormTensorSuffix,
        'queryWeightTensorSuffix': queryWeightTensorSuffix,
        'keyWeightTensorSuffix': keyWeightTensorSuffix,
        'valueWeightTensorSuffix': valueWeightTensorSuffix,
        'outputWeightTensorSuffix': outputWeightTensorSuffix,
        'ffnNormTensorSuffix': ffnNormTensorSuffix,
        'gateWeightTensorSuffix': gateWeightTensorSuffix,
        'upWeightTensorSuffix': upWeightTensorSuffix,
        'downWeightTensorSuffix': downWeightTensorSuffix,
        'finalNormTensorName': finalNormTensorName,
        'lmHeadTensorName': lmHeadTensorName,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'readPosition': readPosition,
        'layerCount': layerCount,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'topK': topK,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(
        result.error ?? 'GGUF next-token generation failed',
      );
    }
    return result;
  }

  Future<GgufGeneratedTokensResult> generateTokens({
    required String path,
    required List<int> tokenIds,
    required int headDim,
    String attnNormTensorSuffix = 'attn_norm.weight',
    String queryWeightTensorSuffix = 'attn_q.weight',
    String keyWeightTensorSuffix = 'attn_k.weight',
    String valueWeightTensorSuffix = 'attn_v.weight',
    String outputWeightTensorSuffix = 'attn_o.weight',
    String ffnNormTensorSuffix = 'ffn_norm.weight',
    String gateWeightTensorSuffix = 'ffn_gate.weight',
    String upWeightTensorSuffix = 'ffn_up.weight',
    String downWeightTensorSuffix = 'ffn_down.weight',
    String finalNormTensorName = 'output_norm.weight',
    String lmHeadTensorName = 'output.weight',
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    int layerCount = -1,
    double ropeTheta = 0,
    int maxNewTokens = 1,
    int topK = 5,
    int maxValues = 16,
  }) async {
    final result = GgufGeneratedTokensResult.fromJson(
      await _invokeJson('generateTokensGguf', path, {
        'tokenIds': tokenIds,
        'attnNormTensorSuffix': attnNormTensorSuffix,
        'queryWeightTensorSuffix': queryWeightTensorSuffix,
        'keyWeightTensorSuffix': keyWeightTensorSuffix,
        'valueWeightTensorSuffix': valueWeightTensorSuffix,
        'outputWeightTensorSuffix': outputWeightTensorSuffix,
        'ffnNormTensorSuffix': ffnNormTensorSuffix,
        'gateWeightTensorSuffix': gateWeightTensorSuffix,
        'upWeightTensorSuffix': upWeightTensorSuffix,
        'downWeightTensorSuffix': downWeightTensorSuffix,
        'finalNormTensorName': finalNormTensorName,
        'lmHeadTensorName': lmHeadTensorName,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'readPosition': readPosition,
        'layerCount': layerCount,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'maxNewTokens': maxNewTokens,
        'topK': topK,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(
        result.error ?? 'GGUF greedy generation failed',
      );
    }
    return result;
  }

  Future<GgufSessionCreateResult> createSession({
    required String path,
    required List<int> tokenIds,
    required int headDim,
    String attnNormTensorSuffix = 'attn_norm.weight',
    String queryWeightTensorSuffix = 'attn_q.weight',
    String keyWeightTensorSuffix = 'attn_k.weight',
    String valueWeightTensorSuffix = 'attn_v.weight',
    String outputWeightTensorSuffix = 'attn_o.weight',
    String ffnNormTensorSuffix = 'ffn_norm.weight',
    String gateWeightTensorSuffix = 'ffn_gate.weight',
    String upWeightTensorSuffix = 'ffn_up.weight',
    String downWeightTensorSuffix = 'ffn_down.weight',
    String finalNormTensorName = 'output_norm.weight',
    String lmHeadTensorName = 'output.weight',
    double epsilon = 0.000001,
    int startPosition = 0,
    int readPosition = -1,
    int layerCount = -1,
    double ropeTheta = 0,
    int maxValues = 16,
  }) async {
    final result = GgufSessionCreateResult.fromJson(
      await _invokeJson('createGgufSession', path, {
        'tokenIds': tokenIds,
        'attnNormTensorSuffix': attnNormTensorSuffix,
        'queryWeightTensorSuffix': queryWeightTensorSuffix,
        'keyWeightTensorSuffix': keyWeightTensorSuffix,
        'valueWeightTensorSuffix': valueWeightTensorSuffix,
        'outputWeightTensorSuffix': outputWeightTensorSuffix,
        'ffnNormTensorSuffix': ffnNormTensorSuffix,
        'gateWeightTensorSuffix': gateWeightTensorSuffix,
        'upWeightTensorSuffix': upWeightTensorSuffix,
        'downWeightTensorSuffix': downWeightTensorSuffix,
        'finalNormTensorName': finalNormTensorName,
        'lmHeadTensorName': lmHeadTensorName,
        'epsilon': epsilon,
        'startPosition': startPosition,
        'readPosition': readPosition,
        'layerCount': layerCount,
        'headDim': headDim,
        'ropeTheta': ropeTheta,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF session creation failed');
    }
    return result;
  }

  Future<GgufSessionDecodeResult> decodeSession({
    required int sessionId,
    int topK = 5,
    int maxValues = 16,
  }) async {
    final result = GgufSessionDecodeResult.fromJson(
      await _invokeJsonArguments('decodeGgufSession', {
        'sessionId': sessionId,
        'topK': topK,
        'maxValues': maxValues,
      }),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF session decode failed');
    }
    return result;
  }

  Future<GgufSessionCloseResult> closeSession(int sessionId) async {
    final result = GgufSessionCloseResult.fromJson(
      await _invokeJsonArguments('closeGgufSession', {'sessionId': sessionId}),
    );
    if (!result.ok) {
      throw GgufRunnerException(result.error ?? 'GGUF session close failed');
    }
    return result;
  }

  Future<Map<String, Object?>> _invokeJson(
    String method,
    String path, [
    Map<String, Object?> extraArguments = const {},
  ]) async {
    return _invokeJsonArguments(method, {'path': path, ...extraArguments});
  }

  Future<Map<String, Object?>> _invokeJsonArguments(
    String method,
    Map<String, Object?> arguments,
  ) async {
    final raw = await _deviceChannel.invokeMethod<String>(method, arguments);
    if (raw == null || raw.trim().isEmpty) {
      throw const GgufRunnerException('Native GGUF runner returned no data');
    }
    final decoded = jsonDecode(raw);
    if (decoded is! Map) {
      throw const GgufRunnerException(
        'Native GGUF runner returned invalid JSON',
      );
    }
    return decoded.cast<String, Object?>();
  }
}

class GgufSessionCreateResult {
  const GgufSessionCreateResult({
    required this.ok,
    required this.sessionId,
    required this.activeSessionCount,
    required this.path,
    required this.architecture,
    required this.prefilled,
    required this.blockCount,
    required this.layerCount,
    required this.promptTokenCount,
    required this.totalTokenCount,
    required this.generatedTokenCount,
    required this.startPosition,
    required this.nextPosition,
    required this.hiddenSize,
    required this.vocabSize,
    required this.headDim,
    required this.queryHeadCount,
    required this.kvHeadCount,
    required this.groupSize,
    required this.kvCacheLayerCount,
    required this.kvCacheTokenCount,
    required this.kvCacheElementCount,
    required this.kvCacheBytesFp32,
    required this.requestedValueCount,
    required this.returnedHiddenValueCount,
    required this.epsilon,
    required this.ropeTheta,
    required this.hiddenMin,
    required this.hiddenMax,
    required this.hiddenMean,
    required this.hiddenL2Norm,
    required this.hiddenChecksum,
    required this.tokenIds,
    required this.allTokenIds,
    required this.qHeadToKvHead,
    required this.layerOutputChecksums,
    required this.hiddenValues,
    this.error,
  });

  factory GgufSessionCreateResult.fromJson(Map<String, Object?> json) {
    return GgufSessionCreateResult(
      ok: _boolValue(json['ok']),
      sessionId: _intValue(json['sessionId']),
      activeSessionCount: _intValue(json['activeSessionCount']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      prefilled: _boolValue(json['prefilled']),
      blockCount: _intValue(json['blockCount']),
      layerCount: _intValue(json['layerCount']),
      promptTokenCount: _intValue(json['promptTokenCount']),
      totalTokenCount: _intValue(json['totalTokenCount']),
      generatedTokenCount: _intValue(json['generatedTokenCount']),
      startPosition: _intValue(json['startPosition']),
      nextPosition: _intValue(json['nextPosition']),
      hiddenSize: _intValue(json['hiddenSize']),
      vocabSize: _intValue(json['vocabSize']),
      headDim: _intValue(json['headDim']),
      queryHeadCount: _intValue(json['queryHeadCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      groupSize: _intValue(json['groupSize']),
      kvCacheLayerCount: _intValue(json['kvCacheLayerCount']),
      kvCacheTokenCount: _intValue(json['kvCacheTokenCount']),
      kvCacheElementCount: _intValue(json['kvCacheElementCount']),
      kvCacheBytesFp32: _intValue(json['kvCacheBytesFp32']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedHiddenValueCount: _intValue(json['returnedHiddenValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      hiddenMin: _doubleValue(json['hiddenMin']),
      hiddenMax: _doubleValue(json['hiddenMax']),
      hiddenMean: _doubleValue(json['hiddenMean']),
      hiddenL2Norm: _doubleValue(json['hiddenL2Norm']),
      hiddenChecksum: _doubleValue(json['hiddenChecksum']),
      tokenIds: _intListValue(json['tokenIds']),
      allTokenIds: _intListValue(json['allTokenIds']),
      qHeadToKvHead: _intListValue(json['qHeadToKvHead']),
      layerOutputChecksums: _doubleListValue(json['layerOutputChecksums']),
      hiddenValues: _doubleListValue(json['hiddenValues']),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final int sessionId;
  final int activeSessionCount;
  final String path;
  final String architecture;
  final bool prefilled;
  final int blockCount;
  final int layerCount;
  final int promptTokenCount;
  final int totalTokenCount;
  final int generatedTokenCount;
  final int startPosition;
  final int nextPosition;
  final int hiddenSize;
  final int vocabSize;
  final int headDim;
  final int queryHeadCount;
  final int kvHeadCount;
  final int groupSize;
  final int kvCacheLayerCount;
  final int kvCacheTokenCount;
  final int kvCacheElementCount;
  final int kvCacheBytesFp32;
  final int requestedValueCount;
  final int returnedHiddenValueCount;
  final double epsilon;
  final double ropeTheta;
  final double hiddenMin;
  final double hiddenMax;
  final double hiddenMean;
  final double hiddenL2Norm;
  final double hiddenChecksum;
  final List<int> tokenIds;
  final List<int> allTokenIds;
  final List<int> qHeadToKvHead;
  final List<double> layerOutputChecksums;
  final List<double> hiddenValues;
  final String? error;
}

class GgufSessionDecodeResult {
  const GgufSessionDecodeResult({
    required this.ok,
    required this.sessionId,
    required this.path,
    required this.architecture,
    required this.decodedPosition,
    required this.nextPosition,
    required this.generatedTokenId,
    required this.generatedTokenCount,
    required this.totalTokenCount,
    required this.hiddenSize,
    required this.vocabSize,
    required this.logitCount,
    required this.kvCacheTokenCount,
    required this.kvCacheElementCount,
    required this.kvCacheBytesFp32,
    required this.requestedTopK,
    required this.returnedTopK,
    required this.requestedValueCount,
    required this.returnedHiddenValueCount,
    required this.returnedNormalizedValueCount,
    required this.returnedLogitValueCount,
    required this.generatedTokenLogit,
    required this.epsilon,
    required this.ropeTheta,
    required this.finalMeanSquare,
    required this.finalInvRms,
    required this.hiddenMin,
    required this.hiddenMax,
    required this.hiddenMean,
    required this.hiddenL2Norm,
    required this.hiddenChecksum,
    required this.normalizedMin,
    required this.normalizedMax,
    required this.normalizedMean,
    required this.normalizedL2Norm,
    required this.normalizedChecksum,
    required this.logitMin,
    required this.logitMax,
    required this.logitMean,
    required this.logitL2Norm,
    required this.logitChecksum,
    required this.generatedTokenIds,
    required this.allTokenIds,
    required this.generatedTokenLogits,
    required this.hiddenValues,
    required this.normalizedValues,
    required this.logitValues,
    required this.topTokenIds,
    required this.topLogits,
    this.error,
  });

  factory GgufSessionDecodeResult.fromJson(Map<String, Object?> json) {
    return GgufSessionDecodeResult(
      ok: _boolValue(json['ok']),
      sessionId: _intValue(json['sessionId']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      decodedPosition: _intValue(json['decodedPosition']),
      nextPosition: _intValue(json['nextPosition']),
      generatedTokenId: _intValue(json['generatedTokenId']),
      generatedTokenCount: _intValue(json['generatedTokenCount']),
      totalTokenCount: _intValue(json['totalTokenCount']),
      hiddenSize: _intValue(json['hiddenSize']),
      vocabSize: _intValue(json['vocabSize']),
      logitCount: _intValue(json['logitCount']),
      kvCacheTokenCount: _intValue(json['kvCacheTokenCount']),
      kvCacheElementCount: _intValue(json['kvCacheElementCount']),
      kvCacheBytesFp32: _intValue(json['kvCacheBytesFp32']),
      requestedTopK: _intValue(json['requestedTopK']),
      returnedTopK: _intValue(json['returnedTopK']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedHiddenValueCount: _intValue(json['returnedHiddenValueCount']),
      returnedNormalizedValueCount: _intValue(
        json['returnedNormalizedValueCount'],
      ),
      returnedLogitValueCount: _intValue(json['returnedLogitValueCount']),
      generatedTokenLogit: _doubleValue(json['generatedTokenLogit']),
      epsilon: _doubleValue(json['epsilon']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      finalMeanSquare: _doubleValue(json['finalMeanSquare']),
      finalInvRms: _doubleValue(json['finalInvRms']),
      hiddenMin: _doubleValue(json['hiddenMin']),
      hiddenMax: _doubleValue(json['hiddenMax']),
      hiddenMean: _doubleValue(json['hiddenMean']),
      hiddenL2Norm: _doubleValue(json['hiddenL2Norm']),
      hiddenChecksum: _doubleValue(json['hiddenChecksum']),
      normalizedMin: _doubleValue(json['normalizedMin']),
      normalizedMax: _doubleValue(json['normalizedMax']),
      normalizedMean: _doubleValue(json['normalizedMean']),
      normalizedL2Norm: _doubleValue(json['normalizedL2Norm']),
      normalizedChecksum: _doubleValue(json['normalizedChecksum']),
      logitMin: _doubleValue(json['logitMin']),
      logitMax: _doubleValue(json['logitMax']),
      logitMean: _doubleValue(json['logitMean']),
      logitL2Norm: _doubleValue(json['logitL2Norm']),
      logitChecksum: _doubleValue(json['logitChecksum']),
      generatedTokenIds: _intListValue(json['generatedTokenIds']),
      allTokenIds: _intListValue(json['allTokenIds']),
      generatedTokenLogits: _doubleListValue(json['generatedTokenLogits']),
      hiddenValues: _doubleListValue(json['hiddenValues']),
      normalizedValues: _doubleListValue(json['normalizedValues']),
      logitValues: _doubleListValue(json['logitValues']),
      topTokenIds: _intListValue(json['topTokenIds']),
      topLogits: _doubleListValue(json['topLogits']),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final int sessionId;
  final String path;
  final String architecture;
  final int decodedPosition;
  final int nextPosition;
  final int generatedTokenId;
  final int generatedTokenCount;
  final int totalTokenCount;
  final int hiddenSize;
  final int vocabSize;
  final int logitCount;
  final int kvCacheTokenCount;
  final int kvCacheElementCount;
  final int kvCacheBytesFp32;
  final int requestedTopK;
  final int returnedTopK;
  final int requestedValueCount;
  final int returnedHiddenValueCount;
  final int returnedNormalizedValueCount;
  final int returnedLogitValueCount;
  final double generatedTokenLogit;
  final double epsilon;
  final double ropeTheta;
  final double finalMeanSquare;
  final double finalInvRms;
  final double hiddenMin;
  final double hiddenMax;
  final double hiddenMean;
  final double hiddenL2Norm;
  final double hiddenChecksum;
  final double normalizedMin;
  final double normalizedMax;
  final double normalizedMean;
  final double normalizedL2Norm;
  final double normalizedChecksum;
  final double logitMin;
  final double logitMax;
  final double logitMean;
  final double logitL2Norm;
  final double logitChecksum;
  final List<int> generatedTokenIds;
  final List<int> allTokenIds;
  final List<double> generatedTokenLogits;
  final List<double> hiddenValues;
  final List<double> normalizedValues;
  final List<double> logitValues;
  final List<int> topTokenIds;
  final List<double> topLogits;
  final String? error;
}

class GgufSessionCloseResult {
  const GgufSessionCloseResult({
    required this.ok,
    required this.sessionId,
    required this.closed,
    required this.activeSessionCount,
    this.error,
  });

  factory GgufSessionCloseResult.fromJson(Map<String, Object?> json) {
    return GgufSessionCloseResult(
      ok: _boolValue(json['ok']),
      sessionId: _intValue(json['sessionId']),
      closed: _boolValue(json['closed']),
      activeSessionCount: _intValue(json['activeSessionCount']),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final int sessionId;
  final bool closed;
  final int activeSessionCount;
  final String? error;
}

class GgufGeneratedTokensResult {
  const GgufGeneratedTokensResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.finalNormTensorName,
    required this.finalNormTensorType,
    required this.lmHeadTensorName,
    required this.lmHeadTensorType,
    required this.cacheReused,
    required this.blockCount,
    required this.layerCount,
    required this.layersVisited,
    required this.tensorsVisited,
    required this.sequenceLength,
    required this.generatedTokenCount,
    required this.totalTokenCount,
    required this.maxNewTokens,
    required this.readPosition,
    required this.selectedTokenId,
    required this.firstGeneratedTokenId,
    required this.lastGeneratedTokenId,
    required this.startPosition,
    required this.hiddenSize,
    required this.vocabSize,
    required this.lmHeadInputLength,
    required this.logitCount,
    required this.headDim,
    required this.queryHeadCount,
    required this.kvHeadCount,
    required this.groupSize,
    required this.kvCacheLayerCount,
    required this.initialKvCacheTokenCount,
    required this.finalKvCacheTokenCount,
    required this.initialKvCacheElementCount,
    required this.finalKvCacheElementCount,
    required this.initialKvCacheBytesFp32,
    required this.finalKvCacheBytesFp32,
    required this.requestedTopK,
    required this.returnedTopK,
    required this.requestedValueCount,
    required this.returnedHiddenValueCount,
    required this.returnedNormalizedValueCount,
    required this.returnedLogitValueCount,
    required this.epsilon,
    required this.ropeTheta,
    required this.scale,
    required this.finalMeanSquare,
    required this.finalInvRms,
    required this.lastTokenLogit,
    required this.hiddenMin,
    required this.hiddenMax,
    required this.hiddenMean,
    required this.hiddenL2Norm,
    required this.hiddenChecksum,
    required this.normalizedMin,
    required this.normalizedMax,
    required this.normalizedMean,
    required this.normalizedL2Norm,
    required this.normalizedChecksum,
    required this.logitMin,
    required this.logitMax,
    required this.logitMean,
    required this.logitL2Norm,
    required this.logitChecksum,
    required this.tokenIds,
    required this.generatedTokenIds,
    required this.allTokenIds,
    required this.generatedTokenLogits,
    required this.qHeadToKvHead,
    required this.layerOutputChecksums,
    required this.hiddenValues,
    required this.normalizedValues,
    required this.logitValues,
    required this.topTokenIds,
    required this.topLogits,
    this.error,
  });

  factory GgufGeneratedTokensResult.fromJson(Map<String, Object?> json) {
    return GgufGeneratedTokensResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      finalNormTensorName: _stringValue(json['finalNormTensorName']),
      finalNormTensorType: _stringValue(json['finalNormTensorType']),
      lmHeadTensorName: _stringValue(json['lmHeadTensorName']),
      lmHeadTensorType: _stringValue(json['lmHeadTensorType']),
      cacheReused: _boolValue(json['cacheReused']),
      blockCount: _intValue(json['blockCount']),
      layerCount: _intValue(json['layerCount']),
      layersVisited: _intValue(json['layersVisited']),
      tensorsVisited: _intValue(json['tensorsVisited']),
      sequenceLength: _intValue(json['sequenceLength']),
      generatedTokenCount: _intValue(json['generatedTokenCount']),
      totalTokenCount: _intValue(json['totalTokenCount']),
      maxNewTokens: _intValue(json['maxNewTokens']),
      readPosition: _intValue(json['readPosition']),
      selectedTokenId: _intValue(json['selectedTokenId']),
      firstGeneratedTokenId: _intValue(json['firstGeneratedTokenId']),
      lastGeneratedTokenId: _intValue(json['lastGeneratedTokenId']),
      startPosition: _intValue(json['startPosition']),
      hiddenSize: _intValue(json['hiddenSize']),
      vocabSize: _intValue(json['vocabSize']),
      lmHeadInputLength: _intValue(json['lmHeadInputLength']),
      logitCount: _intValue(json['logitCount']),
      headDim: _intValue(json['headDim']),
      queryHeadCount: _intValue(json['queryHeadCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      groupSize: _intValue(json['groupSize']),
      kvCacheLayerCount: _intValue(json['kvCacheLayerCount']),
      initialKvCacheTokenCount: _intValue(json['initialKvCacheTokenCount']),
      finalKvCacheTokenCount: _intValue(json['finalKvCacheTokenCount']),
      initialKvCacheElementCount: _intValue(json['initialKvCacheElementCount']),
      finalKvCacheElementCount: _intValue(json['finalKvCacheElementCount']),
      initialKvCacheBytesFp32: _intValue(json['initialKvCacheBytesFp32']),
      finalKvCacheBytesFp32: _intValue(json['finalKvCacheBytesFp32']),
      requestedTopK: _intValue(json['requestedTopK']),
      returnedTopK: _intValue(json['returnedTopK']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedHiddenValueCount: _intValue(json['returnedHiddenValueCount']),
      returnedNormalizedValueCount: _intValue(
        json['returnedNormalizedValueCount'],
      ),
      returnedLogitValueCount: _intValue(json['returnedLogitValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      scale: _doubleValue(json['scale']),
      finalMeanSquare: _doubleValue(json['finalMeanSquare']),
      finalInvRms: _doubleValue(json['finalInvRms']),
      lastTokenLogit: _doubleValue(json['lastTokenLogit']),
      hiddenMin: _doubleValue(json['hiddenMin']),
      hiddenMax: _doubleValue(json['hiddenMax']),
      hiddenMean: _doubleValue(json['hiddenMean']),
      hiddenL2Norm: _doubleValue(json['hiddenL2Norm']),
      hiddenChecksum: _doubleValue(json['hiddenChecksum']),
      normalizedMin: _doubleValue(json['normalizedMin']),
      normalizedMax: _doubleValue(json['normalizedMax']),
      normalizedMean: _doubleValue(json['normalizedMean']),
      normalizedL2Norm: _doubleValue(json['normalizedL2Norm']),
      normalizedChecksum: _doubleValue(json['normalizedChecksum']),
      logitMin: _doubleValue(json['logitMin']),
      logitMax: _doubleValue(json['logitMax']),
      logitMean: _doubleValue(json['logitMean']),
      logitL2Norm: _doubleValue(json['logitL2Norm']),
      logitChecksum: _doubleValue(json['logitChecksum']),
      tokenIds: _listValue(
        json['tokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      generatedTokenIds: _listValue(
        json['generatedTokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      allTokenIds: _listValue(
        json['allTokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      generatedTokenLogits: _doubleListValue(json['generatedTokenLogits']),
      qHeadToKvHead: _listValue(
        json['qHeadToKvHead'],
      ).map((value) => _intValue(value)).toList(growable: false),
      layerOutputChecksums: _doubleListValue(json['layerOutputChecksums']),
      hiddenValues: _doubleListValue(json['hiddenValues']),
      normalizedValues: _doubleListValue(json['normalizedValues']),
      logitValues: _doubleListValue(json['logitValues']),
      topTokenIds: _listValue(
        json['topTokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      topLogits: _doubleListValue(json['topLogits']),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String finalNormTensorName;
  final String finalNormTensorType;
  final String lmHeadTensorName;
  final String lmHeadTensorType;
  final bool cacheReused;
  final int blockCount;
  final int layerCount;
  final int layersVisited;
  final int tensorsVisited;
  final int sequenceLength;
  final int generatedTokenCount;
  final int totalTokenCount;
  final int maxNewTokens;
  final int readPosition;
  final int selectedTokenId;
  final int firstGeneratedTokenId;
  final int lastGeneratedTokenId;
  final int startPosition;
  final int hiddenSize;
  final int vocabSize;
  final int lmHeadInputLength;
  final int logitCount;
  final int headDim;
  final int queryHeadCount;
  final int kvHeadCount;
  final int groupSize;
  final int kvCacheLayerCount;
  final int initialKvCacheTokenCount;
  final int finalKvCacheTokenCount;
  final int initialKvCacheElementCount;
  final int finalKvCacheElementCount;
  final int initialKvCacheBytesFp32;
  final int finalKvCacheBytesFp32;
  final int requestedTopK;
  final int returnedTopK;
  final int requestedValueCount;
  final int returnedHiddenValueCount;
  final int returnedNormalizedValueCount;
  final int returnedLogitValueCount;
  final double epsilon;
  final double ropeTheta;
  final double scale;
  final double finalMeanSquare;
  final double finalInvRms;
  final double lastTokenLogit;
  final double hiddenMin;
  final double hiddenMax;
  final double hiddenMean;
  final double hiddenL2Norm;
  final double hiddenChecksum;
  final double normalizedMin;
  final double normalizedMax;
  final double normalizedMean;
  final double normalizedL2Norm;
  final double normalizedChecksum;
  final double logitMin;
  final double logitMax;
  final double logitMean;
  final double logitL2Norm;
  final double logitChecksum;
  final List<int> tokenIds;
  final List<int> generatedTokenIds;
  final List<int> allTokenIds;
  final List<double> generatedTokenLogits;
  final List<int> qHeadToKvHead;
  final List<double> layerOutputChecksums;
  final List<double> hiddenValues;
  final List<double> normalizedValues;
  final List<double> logitValues;
  final List<int> topTokenIds;
  final List<double> topLogits;
  final String? error;
}

class GgufNextTokenResult {
  const GgufNextTokenResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.finalNormTensorName,
    required this.finalNormTensorType,
    required this.lmHeadTensorName,
    required this.lmHeadTensorType,
    required this.attnNormTensorSuffix,
    required this.queryWeightTensorSuffix,
    required this.keyWeightTensorSuffix,
    required this.valueWeightTensorSuffix,
    required this.outputWeightTensorSuffix,
    required this.ffnNormTensorSuffix,
    required this.gateWeightTensorSuffix,
    required this.upWeightTensorSuffix,
    required this.downWeightTensorSuffix,
    required this.blockCount,
    required this.layerCount,
    required this.layersVisited,
    required this.tensorsVisited,
    required this.sequenceLength,
    required this.readPosition,
    required this.selectedTokenId,
    required this.nextTokenId,
    required this.startPosition,
    required this.hiddenSize,
    required this.vocabSize,
    required this.lmHeadInputLength,
    required this.logitCount,
    required this.headDim,
    required this.queryHeadCount,
    required this.kvHeadCount,
    required this.groupSize,
    required this.kvCacheLayerCount,
    required this.kvCacheElementCount,
    required this.kvCacheBytesFp32,
    required this.requestedTopK,
    required this.returnedTopK,
    required this.requestedValueCount,
    required this.returnedHiddenValueCount,
    required this.returnedNormalizedValueCount,
    required this.returnedLogitValueCount,
    required this.epsilon,
    required this.ropeTheta,
    required this.scale,
    required this.finalMeanSquare,
    required this.finalInvRms,
    required this.nextTokenLogit,
    required this.hiddenMin,
    required this.hiddenMax,
    required this.hiddenMean,
    required this.hiddenL2Norm,
    required this.hiddenChecksum,
    required this.normalizedMin,
    required this.normalizedMax,
    required this.normalizedMean,
    required this.normalizedL2Norm,
    required this.normalizedChecksum,
    required this.logitMin,
    required this.logitMax,
    required this.logitMean,
    required this.logitL2Norm,
    required this.logitChecksum,
    required this.tokenIds,
    required this.qHeadToKvHead,
    required this.layerOutputChecksums,
    required this.hiddenValues,
    required this.normalizedValues,
    required this.logitValues,
    required this.topTokenIds,
    required this.topLogits,
    this.error,
  });

  factory GgufNextTokenResult.fromJson(Map<String, Object?> json) {
    return GgufNextTokenResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      finalNormTensorName: _stringValue(json['finalNormTensorName']),
      finalNormTensorType: _stringValue(json['finalNormTensorType']),
      lmHeadTensorName: _stringValue(json['lmHeadTensorName']),
      lmHeadTensorType: _stringValue(json['lmHeadTensorType']),
      attnNormTensorSuffix: _stringValue(json['attnNormTensorSuffix']),
      queryWeightTensorSuffix: _stringValue(json['queryWeightTensorSuffix']),
      keyWeightTensorSuffix: _stringValue(json['keyWeightTensorSuffix']),
      valueWeightTensorSuffix: _stringValue(json['valueWeightTensorSuffix']),
      outputWeightTensorSuffix: _stringValue(json['outputWeightTensorSuffix']),
      ffnNormTensorSuffix: _stringValue(json['ffnNormTensorSuffix']),
      gateWeightTensorSuffix: _stringValue(json['gateWeightTensorSuffix']),
      upWeightTensorSuffix: _stringValue(json['upWeightTensorSuffix']),
      downWeightTensorSuffix: _stringValue(json['downWeightTensorSuffix']),
      blockCount: _intValue(json['blockCount']),
      layerCount: _intValue(json['layerCount']),
      layersVisited: _intValue(json['layersVisited']),
      tensorsVisited: _intValue(json['tensorsVisited']),
      sequenceLength: _intValue(json['sequenceLength']),
      readPosition: _intValue(json['readPosition']),
      selectedTokenId: _intValue(json['selectedTokenId']),
      nextTokenId: _intValue(json['nextTokenId']),
      startPosition: _intValue(json['startPosition']),
      hiddenSize: _intValue(json['hiddenSize']),
      vocabSize: _intValue(json['vocabSize']),
      lmHeadInputLength: _intValue(json['lmHeadInputLength']),
      logitCount: _intValue(json['logitCount']),
      headDim: _intValue(json['headDim']),
      queryHeadCount: _intValue(json['queryHeadCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      groupSize: _intValue(json['groupSize']),
      kvCacheLayerCount: _intValue(json['kvCacheLayerCount']),
      kvCacheElementCount: _intValue(json['kvCacheElementCount']),
      kvCacheBytesFp32: _intValue(json['kvCacheBytesFp32']),
      requestedTopK: _intValue(json['requestedTopK']),
      returnedTopK: _intValue(json['returnedTopK']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedHiddenValueCount: _intValue(json['returnedHiddenValueCount']),
      returnedNormalizedValueCount: _intValue(
        json['returnedNormalizedValueCount'],
      ),
      returnedLogitValueCount: _intValue(json['returnedLogitValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      scale: _doubleValue(json['scale']),
      finalMeanSquare: _doubleValue(json['finalMeanSquare']),
      finalInvRms: _doubleValue(json['finalInvRms']),
      nextTokenLogit: _doubleValue(json['nextTokenLogit']),
      hiddenMin: _doubleValue(json['hiddenMin']),
      hiddenMax: _doubleValue(json['hiddenMax']),
      hiddenMean: _doubleValue(json['hiddenMean']),
      hiddenL2Norm: _doubleValue(json['hiddenL2Norm']),
      hiddenChecksum: _doubleValue(json['hiddenChecksum']),
      normalizedMin: _doubleValue(json['normalizedMin']),
      normalizedMax: _doubleValue(json['normalizedMax']),
      normalizedMean: _doubleValue(json['normalizedMean']),
      normalizedL2Norm: _doubleValue(json['normalizedL2Norm']),
      normalizedChecksum: _doubleValue(json['normalizedChecksum']),
      logitMin: _doubleValue(json['logitMin']),
      logitMax: _doubleValue(json['logitMax']),
      logitMean: _doubleValue(json['logitMean']),
      logitL2Norm: _doubleValue(json['logitL2Norm']),
      logitChecksum: _doubleValue(json['logitChecksum']),
      tokenIds: _listValue(
        json['tokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      qHeadToKvHead: _listValue(
        json['qHeadToKvHead'],
      ).map((value) => _intValue(value)).toList(growable: false),
      layerOutputChecksums: _doubleListValue(json['layerOutputChecksums']),
      hiddenValues: _doubleListValue(json['hiddenValues']),
      normalizedValues: _doubleListValue(json['normalizedValues']),
      logitValues: _doubleListValue(json['logitValues']),
      topTokenIds: _listValue(
        json['topTokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      topLogits: _doubleListValue(json['topLogits']),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String finalNormTensorName;
  final String finalNormTensorType;
  final String lmHeadTensorName;
  final String lmHeadTensorType;
  final String attnNormTensorSuffix;
  final String queryWeightTensorSuffix;
  final String keyWeightTensorSuffix;
  final String valueWeightTensorSuffix;
  final String outputWeightTensorSuffix;
  final String ffnNormTensorSuffix;
  final String gateWeightTensorSuffix;
  final String upWeightTensorSuffix;
  final String downWeightTensorSuffix;
  final int blockCount;
  final int layerCount;
  final int layersVisited;
  final int tensorsVisited;
  final int sequenceLength;
  final int readPosition;
  final int selectedTokenId;
  final int nextTokenId;
  final int startPosition;
  final int hiddenSize;
  final int vocabSize;
  final int lmHeadInputLength;
  final int logitCount;
  final int headDim;
  final int queryHeadCount;
  final int kvHeadCount;
  final int groupSize;
  final int kvCacheLayerCount;
  final int kvCacheElementCount;
  final int kvCacheBytesFp32;
  final int requestedTopK;
  final int returnedTopK;
  final int requestedValueCount;
  final int returnedHiddenValueCount;
  final int returnedNormalizedValueCount;
  final int returnedLogitValueCount;
  final double epsilon;
  final double ropeTheta;
  final double scale;
  final double finalMeanSquare;
  final double finalInvRms;
  final double nextTokenLogit;
  final double hiddenMin;
  final double hiddenMax;
  final double hiddenMean;
  final double hiddenL2Norm;
  final double hiddenChecksum;
  final double normalizedMin;
  final double normalizedMax;
  final double normalizedMean;
  final double normalizedL2Norm;
  final double normalizedChecksum;
  final double logitMin;
  final double logitMax;
  final double logitMean;
  final double logitL2Norm;
  final double logitChecksum;
  final List<int> tokenIds;
  final List<int> qHeadToKvHead;
  final List<double> layerOutputChecksums;
  final List<double> hiddenValues;
  final List<double> normalizedValues;
  final List<double> logitValues;
  final List<int> topTokenIds;
  final List<double> topLogits;
  final String? error;
}

class GgufTransformerStackResult {
  const GgufTransformerStackResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.attnNormTensorSuffix,
    required this.queryWeightTensorSuffix,
    required this.keyWeightTensorSuffix,
    required this.valueWeightTensorSuffix,
    required this.outputWeightTensorSuffix,
    required this.ffnNormTensorSuffix,
    required this.gateWeightTensorSuffix,
    required this.upWeightTensorSuffix,
    required this.downWeightTensorSuffix,
    required this.blockCount,
    required this.layerCount,
    required this.layersVisited,
    required this.tensorsVisited,
    required this.sequenceLength,
    required this.readPosition,
    required this.selectedTokenId,
    required this.startPosition,
    required this.hiddenSize,
    required this.vocabSize,
    required this.queryWidth,
    required this.keyWidth,
    required this.valueWidth,
    required this.ffnWidth,
    required this.headDim,
    required this.queryHeadCount,
    required this.kvHeadCount,
    required this.groupSize,
    required this.kvCacheLayerCount,
    required this.kvCacheElementCount,
    required this.kvCacheBytesFp32,
    required this.requestedValueCount,
    required this.returnedOutputValueCount,
    required this.returnedAttentionValueCount,
    required this.returnedFfnValueCount,
    required this.returnedScoreValueCount,
    required this.epsilon,
    required this.ropeTheta,
    required this.scale,
    required this.lastLayerAttnMeanSquare,
    required this.lastLayerAttnInvRms,
    required this.lastLayerFfnMeanSquare,
    required this.lastLayerFfnInvRms,
    required this.outputMin,
    required this.outputMax,
    required this.outputMean,
    required this.outputL2Norm,
    required this.outputChecksum,
    required this.tokenIds,
    required this.qHeadToKvHead,
    required this.layerOutputChecksums,
    required this.lastLayerQueryValues,
    required this.lastLayerAttentionValues,
    required this.lastLayerAttentionProjectedValues,
    required this.lastLayerPostAttentionValues,
    required this.lastLayerGateValues,
    required this.lastLayerUpValues,
    required this.lastLayerFfnHiddenValues,
    required this.lastLayerFfnOutputValues,
    required this.outputValues,
    required this.lastLayerFirstHeadScores,
    required this.lastLayerFirstHeadProbabilities,
    this.error,
  });

  factory GgufTransformerStackResult.fromJson(Map<String, Object?> json) {
    return GgufTransformerStackResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      attnNormTensorSuffix: _stringValue(json['attnNormTensorSuffix']),
      queryWeightTensorSuffix: _stringValue(json['queryWeightTensorSuffix']),
      keyWeightTensorSuffix: _stringValue(json['keyWeightTensorSuffix']),
      valueWeightTensorSuffix: _stringValue(json['valueWeightTensorSuffix']),
      outputWeightTensorSuffix: _stringValue(json['outputWeightTensorSuffix']),
      ffnNormTensorSuffix: _stringValue(json['ffnNormTensorSuffix']),
      gateWeightTensorSuffix: _stringValue(json['gateWeightTensorSuffix']),
      upWeightTensorSuffix: _stringValue(json['upWeightTensorSuffix']),
      downWeightTensorSuffix: _stringValue(json['downWeightTensorSuffix']),
      blockCount: _intValue(json['blockCount']),
      layerCount: _intValue(json['layerCount']),
      layersVisited: _intValue(json['layersVisited']),
      tensorsVisited: _intValue(json['tensorsVisited']),
      sequenceLength: _intValue(json['sequenceLength']),
      readPosition: _intValue(json['readPosition']),
      selectedTokenId: _intValue(json['selectedTokenId']),
      startPosition: _intValue(json['startPosition']),
      hiddenSize: _intValue(json['hiddenSize']),
      vocabSize: _intValue(json['vocabSize']),
      queryWidth: _intValue(json['queryWidth']),
      keyWidth: _intValue(json['keyWidth']),
      valueWidth: _intValue(json['valueWidth']),
      ffnWidth: _intValue(json['ffnWidth']),
      headDim: _intValue(json['headDim']),
      queryHeadCount: _intValue(json['queryHeadCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      groupSize: _intValue(json['groupSize']),
      kvCacheLayerCount: _intValue(json['kvCacheLayerCount']),
      kvCacheElementCount: _intValue(json['kvCacheElementCount']),
      kvCacheBytesFp32: _intValue(json['kvCacheBytesFp32']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedOutputValueCount: _intValue(json['returnedOutputValueCount']),
      returnedAttentionValueCount: _intValue(
        json['returnedAttentionValueCount'],
      ),
      returnedFfnValueCount: _intValue(json['returnedFfnValueCount']),
      returnedScoreValueCount: _intValue(json['returnedScoreValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      scale: _doubleValue(json['scale']),
      lastLayerAttnMeanSquare: _doubleValue(json['lastLayerAttnMeanSquare']),
      lastLayerAttnInvRms: _doubleValue(json['lastLayerAttnInvRms']),
      lastLayerFfnMeanSquare: _doubleValue(json['lastLayerFfnMeanSquare']),
      lastLayerFfnInvRms: _doubleValue(json['lastLayerFfnInvRms']),
      outputMin: _doubleValue(json['outputMin']),
      outputMax: _doubleValue(json['outputMax']),
      outputMean: _doubleValue(json['outputMean']),
      outputL2Norm: _doubleValue(json['outputL2Norm']),
      outputChecksum: _doubleValue(json['outputChecksum']),
      tokenIds: _listValue(
        json['tokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      qHeadToKvHead: _listValue(
        json['qHeadToKvHead'],
      ).map((value) => _intValue(value)).toList(growable: false),
      layerOutputChecksums: _doubleListValue(json['layerOutputChecksums']),
      lastLayerQueryValues: _doubleListValue(json['lastLayerQueryValues']),
      lastLayerAttentionValues: _doubleListValue(
        json['lastLayerAttentionValues'],
      ),
      lastLayerAttentionProjectedValues: _doubleListValue(
        json['lastLayerAttentionProjectedValues'],
      ),
      lastLayerPostAttentionValues: _doubleListValue(
        json['lastLayerPostAttentionValues'],
      ),
      lastLayerGateValues: _doubleListValue(json['lastLayerGateValues']),
      lastLayerUpValues: _doubleListValue(json['lastLayerUpValues']),
      lastLayerFfnHiddenValues: _doubleListValue(
        json['lastLayerFfnHiddenValues'],
      ),
      lastLayerFfnOutputValues: _doubleListValue(
        json['lastLayerFfnOutputValues'],
      ),
      outputValues: _doubleListValue(json['outputValues']),
      lastLayerFirstHeadScores: _doubleListValue(
        json['lastLayerFirstHeadScores'],
      ),
      lastLayerFirstHeadProbabilities: _doubleListValue(
        json['lastLayerFirstHeadProbabilities'],
      ),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String attnNormTensorSuffix;
  final String queryWeightTensorSuffix;
  final String keyWeightTensorSuffix;
  final String valueWeightTensorSuffix;
  final String outputWeightTensorSuffix;
  final String ffnNormTensorSuffix;
  final String gateWeightTensorSuffix;
  final String upWeightTensorSuffix;
  final String downWeightTensorSuffix;
  final int blockCount;
  final int layerCount;
  final int layersVisited;
  final int tensorsVisited;
  final int sequenceLength;
  final int readPosition;
  final int selectedTokenId;
  final int startPosition;
  final int hiddenSize;
  final int vocabSize;
  final int queryWidth;
  final int keyWidth;
  final int valueWidth;
  final int ffnWidth;
  final int headDim;
  final int queryHeadCount;
  final int kvHeadCount;
  final int groupSize;
  final int kvCacheLayerCount;
  final int kvCacheElementCount;
  final int kvCacheBytesFp32;
  final int requestedValueCount;
  final int returnedOutputValueCount;
  final int returnedAttentionValueCount;
  final int returnedFfnValueCount;
  final int returnedScoreValueCount;
  final double epsilon;
  final double ropeTheta;
  final double scale;
  final double lastLayerAttnMeanSquare;
  final double lastLayerAttnInvRms;
  final double lastLayerFfnMeanSquare;
  final double lastLayerFfnInvRms;
  final double outputMin;
  final double outputMax;
  final double outputMean;
  final double outputL2Norm;
  final double outputChecksum;
  final List<int> tokenIds;
  final List<int> qHeadToKvHead;
  final List<double> layerOutputChecksums;
  final List<double> lastLayerQueryValues;
  final List<double> lastLayerAttentionValues;
  final List<double> lastLayerAttentionProjectedValues;
  final List<double> lastLayerPostAttentionValues;
  final List<double> lastLayerGateValues;
  final List<double> lastLayerUpValues;
  final List<double> lastLayerFfnHiddenValues;
  final List<double> lastLayerFfnOutputValues;
  final List<double> outputValues;
  final List<double> lastLayerFirstHeadScores;
  final List<double> lastLayerFirstHeadProbabilities;
  final String? error;
}

class GgufTransformerLayerResult {
  const GgufTransformerLayerResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.queryTokenId,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.attnNormTensorName,
    required this.attnNormTensorType,
    required this.queryWeightTensorName,
    required this.queryWeightTensorType,
    required this.keyWeightTensorName,
    required this.keyWeightTensorType,
    required this.valueWeightTensorName,
    required this.valueWeightTensorType,
    required this.outputWeightTensorName,
    required this.outputWeightTensorType,
    required this.ffnNormTensorName,
    required this.ffnNormTensorType,
    required this.gateWeightTensorName,
    required this.gateWeightTensorType,
    required this.upWeightTensorName,
    required this.upWeightTensorType,
    required this.downWeightTensorName,
    required this.downWeightTensorType,
    required this.sequenceLength,
    required this.attentionLength,
    required this.embeddingLength,
    required this.vocabSize,
    required this.queryWidth,
    required this.keyWidth,
    required this.valueWidth,
    required this.attentionOutputWidth,
    required this.hiddenSize,
    required this.ffnWidth,
    required this.headDim,
    required this.queryHeadCount,
    required this.kvHeadCount,
    required this.groupSize,
    required this.startPosition,
    required this.queryPosition,
    required this.readPosition,
    required this.requestedValueCount,
    required this.returnedOutputValueCount,
    required this.returnedAttentionValueCount,
    required this.returnedFfnValueCount,
    required this.returnedScoreValueCount,
    required this.epsilon,
    required this.attnMeanSquare,
    required this.attnInvRms,
    required this.ffnMeanSquare,
    required this.ffnInvRms,
    required this.ropeTheta,
    required this.scale,
    required this.outputMin,
    required this.outputMax,
    required this.outputMean,
    required this.outputL2Norm,
    required this.outputChecksum,
    required this.tokenIds,
    required this.qHeadToKvHead,
    required this.queryValues,
    required this.attentionValues,
    required this.attentionProjectedValues,
    required this.postAttentionValues,
    required this.gateValues,
    required this.upValues,
    required this.ffnHiddenValues,
    required this.ffnOutputValues,
    required this.outputValues,
    required this.firstHeadScores,
    required this.firstHeadProbabilities,
    this.error,
  });

  factory GgufTransformerLayerResult.fromJson(Map<String, Object?> json) {
    return GgufTransformerLayerResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      queryTokenId: _intValue(json['queryTokenId']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      attnNormTensorName: _stringValue(json['attnNormTensorName']),
      attnNormTensorType: _stringValue(json['attnNormTensorType']),
      queryWeightTensorName: _stringValue(json['queryWeightTensorName']),
      queryWeightTensorType: _stringValue(json['queryWeightTensorType']),
      keyWeightTensorName: _stringValue(json['keyWeightTensorName']),
      keyWeightTensorType: _stringValue(json['keyWeightTensorType']),
      valueWeightTensorName: _stringValue(json['valueWeightTensorName']),
      valueWeightTensorType: _stringValue(json['valueWeightTensorType']),
      outputWeightTensorName: _stringValue(json['outputWeightTensorName']),
      outputWeightTensorType: _stringValue(json['outputWeightTensorType']),
      ffnNormTensorName: _stringValue(json['ffnNormTensorName']),
      ffnNormTensorType: _stringValue(json['ffnNormTensorType']),
      gateWeightTensorName: _stringValue(json['gateWeightTensorName']),
      gateWeightTensorType: _stringValue(json['gateWeightTensorType']),
      upWeightTensorName: _stringValue(json['upWeightTensorName']),
      upWeightTensorType: _stringValue(json['upWeightTensorType']),
      downWeightTensorName: _stringValue(json['downWeightTensorName']),
      downWeightTensorType: _stringValue(json['downWeightTensorType']),
      sequenceLength: _intValue(json['sequenceLength']),
      attentionLength: _intValue(json['attentionLength']),
      embeddingLength: _intValue(json['embeddingLength']),
      vocabSize: _intValue(json['vocabSize']),
      queryWidth: _intValue(json['queryWidth']),
      keyWidth: _intValue(json['keyWidth']),
      valueWidth: _intValue(json['valueWidth']),
      attentionOutputWidth: _intValue(json['attentionOutputWidth']),
      hiddenSize: _intValue(json['hiddenSize']),
      ffnWidth: _intValue(json['ffnWidth']),
      headDim: _intValue(json['headDim']),
      queryHeadCount: _intValue(json['queryHeadCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      groupSize: _intValue(json['groupSize']),
      startPosition: _intValue(json['startPosition']),
      queryPosition: _intValue(json['queryPosition']),
      readPosition: _intValue(json['readPosition']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedOutputValueCount: _intValue(json['returnedOutputValueCount']),
      returnedAttentionValueCount: _intValue(
        json['returnedAttentionValueCount'],
      ),
      returnedFfnValueCount: _intValue(json['returnedFfnValueCount']),
      returnedScoreValueCount: _intValue(json['returnedScoreValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      attnMeanSquare: _doubleValue(json['attnMeanSquare']),
      attnInvRms: _doubleValue(json['attnInvRms']),
      ffnMeanSquare: _doubleValue(json['ffnMeanSquare']),
      ffnInvRms: _doubleValue(json['ffnInvRms']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      scale: _doubleValue(json['scale']),
      outputMin: _doubleValue(json['outputMin']),
      outputMax: _doubleValue(json['outputMax']),
      outputMean: _doubleValue(json['outputMean']),
      outputL2Norm: _doubleValue(json['outputL2Norm']),
      outputChecksum: _doubleValue(json['outputChecksum']),
      tokenIds: _listValue(
        json['tokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      qHeadToKvHead: _listValue(
        json['qHeadToKvHead'],
      ).map((value) => _intValue(value)).toList(growable: false),
      queryValues: _doubleListValue(json['queryValues']),
      attentionValues: _doubleListValue(json['attentionValues']),
      attentionProjectedValues: _doubleListValue(
        json['attentionProjectedValues'],
      ),
      postAttentionValues: _doubleListValue(json['postAttentionValues']),
      gateValues: _doubleListValue(json['gateValues']),
      upValues: _doubleListValue(json['upValues']),
      ffnHiddenValues: _doubleListValue(json['ffnHiddenValues']),
      ffnOutputValues: _doubleListValue(json['ffnOutputValues']),
      outputValues: _doubleListValue(json['outputValues']),
      firstHeadScores: _doubleListValue(json['firstHeadScores']),
      firstHeadProbabilities: _doubleListValue(json['firstHeadProbabilities']),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final int queryTokenId;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String attnNormTensorName;
  final String attnNormTensorType;
  final String queryWeightTensorName;
  final String queryWeightTensorType;
  final String keyWeightTensorName;
  final String keyWeightTensorType;
  final String valueWeightTensorName;
  final String valueWeightTensorType;
  final String outputWeightTensorName;
  final String outputWeightTensorType;
  final String ffnNormTensorName;
  final String ffnNormTensorType;
  final String gateWeightTensorName;
  final String gateWeightTensorType;
  final String upWeightTensorName;
  final String upWeightTensorType;
  final String downWeightTensorName;
  final String downWeightTensorType;
  final int sequenceLength;
  final int attentionLength;
  final int embeddingLength;
  final int vocabSize;
  final int queryWidth;
  final int keyWidth;
  final int valueWidth;
  final int attentionOutputWidth;
  final int hiddenSize;
  final int ffnWidth;
  final int headDim;
  final int queryHeadCount;
  final int kvHeadCount;
  final int groupSize;
  final int startPosition;
  final int queryPosition;
  final int readPosition;
  final int requestedValueCount;
  final int returnedOutputValueCount;
  final int returnedAttentionValueCount;
  final int returnedFfnValueCount;
  final int returnedScoreValueCount;
  final double epsilon;
  final double attnMeanSquare;
  final double attnInvRms;
  final double ffnMeanSquare;
  final double ffnInvRms;
  final double ropeTheta;
  final double scale;
  final double outputMin;
  final double outputMax;
  final double outputMean;
  final double outputL2Norm;
  final double outputChecksum;
  final List<int> tokenIds;
  final List<int> qHeadToKvHead;
  final List<double> queryValues;
  final List<double> attentionValues;
  final List<double> attentionProjectedValues;
  final List<double> postAttentionValues;
  final List<double> gateValues;
  final List<double> upValues;
  final List<double> ffnHiddenValues;
  final List<double> ffnOutputValues;
  final List<double> outputValues;
  final List<double> firstHeadScores;
  final List<double> firstHeadProbabilities;
  final String? error;
}

class GgufAttentionResult {
  const GgufAttentionResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.queryTokenId,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.normTensorName,
    required this.normTensorType,
    required this.queryWeightTensorName,
    required this.queryWeightTensorType,
    required this.keyWeightTensorName,
    required this.keyWeightTensorType,
    required this.valueWeightTensorName,
    required this.valueWeightTensorType,
    required this.sequenceLength,
    required this.attentionLength,
    required this.embeddingLength,
    required this.vocabSize,
    required this.queryWidth,
    required this.keyWidth,
    required this.valueWidth,
    required this.headDim,
    required this.headIndex,
    required this.queryHeadCount,
    required this.kvHeadCount,
    required this.startPosition,
    required this.queryPosition,
    required this.readPosition,
    required this.requestedValueCount,
    required this.returnedOutputValueCount,
    required this.returnedScoreValueCount,
    required this.epsilon,
    required this.queryMeanSquare,
    required this.queryInvRms,
    required this.ropeTheta,
    required this.scale,
    required this.maxScore,
    required this.softmaxDenominator,
    required this.outputMin,
    required this.outputMax,
    required this.outputMean,
    required this.outputL2Norm,
    required this.outputChecksum,
    required this.tokenIds,
    required this.queryValues,
    required this.attentionScores,
    required this.attentionProbabilities,
    required this.outputValues,
    this.error,
  });

  factory GgufAttentionResult.fromJson(Map<String, Object?> json) {
    return GgufAttentionResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      queryTokenId: _intValue(json['queryTokenId']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      normTensorName: _stringValue(json['normTensorName']),
      normTensorType: _stringValue(json['normTensorType']),
      queryWeightTensorName: _stringValue(json['queryWeightTensorName']),
      queryWeightTensorType: _stringValue(json['queryWeightTensorType']),
      keyWeightTensorName: _stringValue(json['keyWeightTensorName']),
      keyWeightTensorType: _stringValue(json['keyWeightTensorType']),
      valueWeightTensorName: _stringValue(json['valueWeightTensorName']),
      valueWeightTensorType: _stringValue(json['valueWeightTensorType']),
      sequenceLength: _intValue(json['sequenceLength']),
      attentionLength: _intValue(json['attentionLength']),
      embeddingLength: _intValue(json['embeddingLength']),
      vocabSize: _intValue(json['vocabSize']),
      queryWidth: _intValue(json['queryWidth']),
      keyWidth: _intValue(json['keyWidth']),
      valueWidth: _intValue(json['valueWidth']),
      headDim: _intValue(json['headDim']),
      headIndex: _intValue(json['headIndex']),
      queryHeadCount: _intValue(json['queryHeadCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      startPosition: _intValue(json['startPosition']),
      queryPosition: _intValue(json['queryPosition']),
      readPosition: _intValue(json['readPosition']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedOutputValueCount: _intValue(json['returnedOutputValueCount']),
      returnedScoreValueCount: _intValue(json['returnedScoreValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      queryMeanSquare: _doubleValue(json['queryMeanSquare']),
      queryInvRms: _doubleValue(json['queryInvRms']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      scale: _doubleValue(json['scale']),
      maxScore: _doubleValue(json['maxScore']),
      softmaxDenominator: _doubleValue(json['softmaxDenominator']),
      outputMin: _doubleValue(json['outputMin']),
      outputMax: _doubleValue(json['outputMax']),
      outputMean: _doubleValue(json['outputMean']),
      outputL2Norm: _doubleValue(json['outputL2Norm']),
      outputChecksum: _doubleValue(json['outputChecksum']),
      tokenIds: _listValue(
        json['tokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      queryValues: _listValue(
        json['queryValues'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      attentionScores: _listValue(
        json['attentionScores'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      attentionProbabilities: _listValue(
        json['attentionProbabilities'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      outputValues: _listValue(
        json['outputValues'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final int queryTokenId;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String normTensorName;
  final String normTensorType;
  final String queryWeightTensorName;
  final String queryWeightTensorType;
  final String keyWeightTensorName;
  final String keyWeightTensorType;
  final String valueWeightTensorName;
  final String valueWeightTensorType;
  final int sequenceLength;
  final int attentionLength;
  final int embeddingLength;
  final int vocabSize;
  final int queryWidth;
  final int keyWidth;
  final int valueWidth;
  final int headDim;
  final int headIndex;
  final int queryHeadCount;
  final int kvHeadCount;
  final int startPosition;
  final int queryPosition;
  final int readPosition;
  final int requestedValueCount;
  final int returnedOutputValueCount;
  final int returnedScoreValueCount;
  final double epsilon;
  final double queryMeanSquare;
  final double queryInvRms;
  final double ropeTheta;
  final double scale;
  final double maxScore;
  final double softmaxDenominator;
  final double outputMin;
  final double outputMax;
  final double outputMean;
  final double outputL2Norm;
  final double outputChecksum;
  final List<int> tokenIds;
  final List<double> queryValues;
  final List<double> attentionScores;
  final List<double> attentionProbabilities;
  final List<double> outputValues;
  final String? error;
}

class GgufMultiHeadAttentionResult {
  const GgufMultiHeadAttentionResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.queryTokenId,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.normTensorName,
    required this.normTensorType,
    required this.queryWeightTensorName,
    required this.queryWeightTensorType,
    required this.keyWeightTensorName,
    required this.keyWeightTensorType,
    required this.valueWeightTensorName,
    required this.valueWeightTensorType,
    required this.sequenceLength,
    required this.attentionLength,
    required this.embeddingLength,
    required this.vocabSize,
    required this.queryWidth,
    required this.keyWidth,
    required this.valueWidth,
    required this.headDim,
    required this.queryHeadCount,
    required this.kvHeadCount,
    required this.groupSize,
    required this.startPosition,
    required this.queryPosition,
    required this.readPosition,
    required this.requestedValueCount,
    required this.returnedOutputValueCount,
    required this.returnedScoreValueCount,
    required this.epsilon,
    required this.queryMeanSquare,
    required this.queryInvRms,
    required this.ropeTheta,
    required this.scale,
    required this.outputMin,
    required this.outputMax,
    required this.outputMean,
    required this.outputL2Norm,
    required this.outputChecksum,
    required this.tokenIds,
    required this.qHeadToKvHead,
    required this.queryValues,
    required this.firstHeadScores,
    required this.firstHeadProbabilities,
    required this.outputValues,
    this.error,
  });

  factory GgufMultiHeadAttentionResult.fromJson(Map<String, Object?> json) {
    return GgufMultiHeadAttentionResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      queryTokenId: _intValue(json['queryTokenId']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      normTensorName: _stringValue(json['normTensorName']),
      normTensorType: _stringValue(json['normTensorType']),
      queryWeightTensorName: _stringValue(json['queryWeightTensorName']),
      queryWeightTensorType: _stringValue(json['queryWeightTensorType']),
      keyWeightTensorName: _stringValue(json['keyWeightTensorName']),
      keyWeightTensorType: _stringValue(json['keyWeightTensorType']),
      valueWeightTensorName: _stringValue(json['valueWeightTensorName']),
      valueWeightTensorType: _stringValue(json['valueWeightTensorType']),
      sequenceLength: _intValue(json['sequenceLength']),
      attentionLength: _intValue(json['attentionLength']),
      embeddingLength: _intValue(json['embeddingLength']),
      vocabSize: _intValue(json['vocabSize']),
      queryWidth: _intValue(json['queryWidth']),
      keyWidth: _intValue(json['keyWidth']),
      valueWidth: _intValue(json['valueWidth']),
      headDim: _intValue(json['headDim']),
      queryHeadCount: _intValue(json['queryHeadCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      groupSize: _intValue(json['groupSize']),
      startPosition: _intValue(json['startPosition']),
      queryPosition: _intValue(json['queryPosition']),
      readPosition: _intValue(json['readPosition']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedOutputValueCount: _intValue(json['returnedOutputValueCount']),
      returnedScoreValueCount: _intValue(json['returnedScoreValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      queryMeanSquare: _doubleValue(json['queryMeanSquare']),
      queryInvRms: _doubleValue(json['queryInvRms']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      scale: _doubleValue(json['scale']),
      outputMin: _doubleValue(json['outputMin']),
      outputMax: _doubleValue(json['outputMax']),
      outputMean: _doubleValue(json['outputMean']),
      outputL2Norm: _doubleValue(json['outputL2Norm']),
      outputChecksum: _doubleValue(json['outputChecksum']),
      tokenIds: _listValue(
        json['tokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      qHeadToKvHead: _listValue(
        json['qHeadToKvHead'],
      ).map((value) => _intValue(value)).toList(growable: false),
      queryValues: _listValue(
        json['queryValues'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      firstHeadScores: _listValue(
        json['firstHeadScores'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      firstHeadProbabilities: _listValue(
        json['firstHeadProbabilities'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      outputValues: _listValue(
        json['outputValues'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final int queryTokenId;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String normTensorName;
  final String normTensorType;
  final String queryWeightTensorName;
  final String queryWeightTensorType;
  final String keyWeightTensorName;
  final String keyWeightTensorType;
  final String valueWeightTensorName;
  final String valueWeightTensorType;
  final int sequenceLength;
  final int attentionLength;
  final int embeddingLength;
  final int vocabSize;
  final int queryWidth;
  final int keyWidth;
  final int valueWidth;
  final int headDim;
  final int queryHeadCount;
  final int kvHeadCount;
  final int groupSize;
  final int startPosition;
  final int queryPosition;
  final int readPosition;
  final int requestedValueCount;
  final int returnedOutputValueCount;
  final int returnedScoreValueCount;
  final double epsilon;
  final double queryMeanSquare;
  final double queryInvRms;
  final double ropeTheta;
  final double scale;
  final double outputMin;
  final double outputMax;
  final double outputMean;
  final double outputL2Norm;
  final double outputChecksum;
  final List<int> tokenIds;
  final List<int> qHeadToKvHead;
  final List<double> queryValues;
  final List<double> firstHeadScores;
  final List<double> firstHeadProbabilities;
  final List<double> outputValues;
  final String? error;
}

class GgufKvCacheResult {
  const GgufKvCacheResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.normTensorName,
    required this.normTensorType,
    required this.keyWeightTensorName,
    required this.keyWeightTensorType,
    required this.valueWeightTensorName,
    required this.valueWeightTensorType,
    required this.sequenceLength,
    required this.embeddingLength,
    required this.vocabSize,
    required this.keyWidth,
    required this.valueWidth,
    required this.headDim,
    required this.kvHeadCount,
    required this.cacheElementCount,
    required this.cacheBytesFp32,
    required this.startPosition,
    required this.readPosition,
    required this.readAbsolutePosition,
    required this.requestedValueCount,
    required this.returnedKeyValueCount,
    required this.returnedValueValueCount,
    required this.epsilon,
    required this.lastMeanSquare,
    required this.lastInvRms,
    required this.ropeTheta,
    required this.keyMin,
    required this.keyMax,
    required this.keyMean,
    required this.keyL2Norm,
    required this.keyChecksum,
    required this.valueMin,
    required this.valueMax,
    required this.valueMean,
    required this.valueL2Norm,
    required this.valueChecksum,
    required this.tokenIds,
    required this.keyWeightShape,
    required this.valueWeightShape,
    required this.keyValues,
    required this.valueValues,
    this.error,
  });

  factory GgufKvCacheResult.fromJson(Map<String, Object?> json) {
    return GgufKvCacheResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      normTensorName: _stringValue(json['normTensorName']),
      normTensorType: _stringValue(json['normTensorType']),
      keyWeightTensorName: _stringValue(json['keyWeightTensorName']),
      keyWeightTensorType: _stringValue(json['keyWeightTensorType']),
      valueWeightTensorName: _stringValue(json['valueWeightTensorName']),
      valueWeightTensorType: _stringValue(json['valueWeightTensorType']),
      sequenceLength: _intValue(json['sequenceLength']),
      embeddingLength: _intValue(json['embeddingLength']),
      vocabSize: _intValue(json['vocabSize']),
      keyWidth: _intValue(json['keyWidth']),
      valueWidth: _intValue(json['valueWidth']),
      headDim: _intValue(json['headDim']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      cacheElementCount: _intValue(json['cacheElementCount']),
      cacheBytesFp32: _intValue(json['cacheBytesFp32']),
      startPosition: _intValue(json['startPosition']),
      readPosition: _intValue(json['readPosition']),
      readAbsolutePosition: _intValue(json['readAbsolutePosition']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedKeyValueCount: _intValue(json['returnedKeyValueCount']),
      returnedValueValueCount: _intValue(json['returnedValueValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      lastMeanSquare: _doubleValue(json['lastMeanSquare']),
      lastInvRms: _doubleValue(json['lastInvRms']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      keyMin: _doubleValue(json['keyMin']),
      keyMax: _doubleValue(json['keyMax']),
      keyMean: _doubleValue(json['keyMean']),
      keyL2Norm: _doubleValue(json['keyL2Norm']),
      keyChecksum: _doubleValue(json['keyChecksum']),
      valueMin: _doubleValue(json['valueMin']),
      valueMax: _doubleValue(json['valueMax']),
      valueMean: _doubleValue(json['valueMean']),
      valueL2Norm: _doubleValue(json['valueL2Norm']),
      valueChecksum: _doubleValue(json['valueChecksum']),
      tokenIds: _listValue(
        json['tokenIds'],
      ).map((value) => _intValue(value)).toList(growable: false),
      keyWeightShape: _listValue(
        json['keyWeightShape'],
      ).map((value) => _intValue(value)).toList(growable: false),
      valueWeightShape: _listValue(
        json['valueWeightShape'],
      ).map((value) => _intValue(value)).toList(growable: false),
      keyValues: _listValue(
        json['keyValues'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      valueValues: _listValue(
        json['valueValues'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String normTensorName;
  final String normTensorType;
  final String keyWeightTensorName;
  final String keyWeightTensorType;
  final String valueWeightTensorName;
  final String valueWeightTensorType;
  final int sequenceLength;
  final int embeddingLength;
  final int vocabSize;
  final int keyWidth;
  final int valueWidth;
  final int headDim;
  final int kvHeadCount;
  final int cacheElementCount;
  final int cacheBytesFp32;
  final int startPosition;
  final int readPosition;
  final int readAbsolutePosition;
  final int requestedValueCount;
  final int returnedKeyValueCount;
  final int returnedValueValueCount;
  final double epsilon;
  final double lastMeanSquare;
  final double lastInvRms;
  final double ropeTheta;
  final double keyMin;
  final double keyMax;
  final double keyMean;
  final double keyL2Norm;
  final double keyChecksum;
  final double valueMin;
  final double valueMax;
  final double valueMean;
  final double valueL2Norm;
  final double valueChecksum;
  final List<int> tokenIds;
  final List<int> keyWeightShape;
  final List<int> valueWeightShape;
  final List<double> keyValues;
  final List<double> valueValues;
  final String? error;
}

class GgufRopeResult {
  const GgufRopeResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.tokenId,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.normTensorName,
    required this.normTensorType,
    required this.weightTensorName,
    required this.weightTensorType,
    required this.inputLength,
    required this.outputLength,
    required this.vocabSize,
    required this.position,
    required this.headDim,
    required this.headCount,
    required this.rotatedPairCount,
    required this.requestedValueCount,
    required this.returnedValueCount,
    required this.epsilon,
    required this.meanSquare,
    required this.invRms,
    required this.ropeTheta,
    required this.min,
    required this.max,
    required this.mean,
    required this.l2Norm,
    required this.checksum,
    required this.weightShape,
    required this.values,
    this.error,
  });

  factory GgufRopeResult.fromJson(Map<String, Object?> json) {
    return GgufRopeResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      tokenId: _intValue(json['tokenId']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      normTensorName: _stringValue(json['normTensorName']),
      normTensorType: _stringValue(json['normTensorType']),
      weightTensorName: _stringValue(json['weightTensorName']),
      weightTensorType: _stringValue(json['weightTensorType']),
      inputLength: _intValue(json['inputLength']),
      outputLength: _intValue(json['outputLength']),
      vocabSize: _intValue(json['vocabSize']),
      position: _intValue(json['position']),
      headDim: _intValue(json['headDim']),
      headCount: _intValue(json['headCount']),
      rotatedPairCount: _intValue(json['rotatedPairCount']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedValueCount: _intValue(json['returnedValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      meanSquare: _doubleValue(json['meanSquare']),
      invRms: _doubleValue(json['invRms']),
      ropeTheta: _doubleValue(json['ropeTheta']),
      min: _doubleValue(json['min']),
      max: _doubleValue(json['max']),
      mean: _doubleValue(json['mean']),
      l2Norm: _doubleValue(json['l2Norm']),
      checksum: _doubleValue(json['checksum']),
      weightShape: _listValue(
        json['weightShape'],
      ).map((value) => _intValue(value)).toList(growable: false),
      values: _listValue(
        json['values'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final int tokenId;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String normTensorName;
  final String normTensorType;
  final String weightTensorName;
  final String weightTensorType;
  final int inputLength;
  final int outputLength;
  final int vocabSize;
  final int position;
  final int headDim;
  final int headCount;
  final int rotatedPairCount;
  final int requestedValueCount;
  final int returnedValueCount;
  final double epsilon;
  final double meanSquare;
  final double invRms;
  final double ropeTheta;
  final double min;
  final double max;
  final double mean;
  final double l2Norm;
  final double checksum;
  final List<int> weightShape;
  final List<double> values;
  final String? error;
}

class GgufMatVecResult {
  const GgufMatVecResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.tokenId,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.normTensorName,
    required this.normTensorType,
    required this.weightTensorName,
    required this.weightTensorType,
    required this.inputLength,
    required this.outputLength,
    required this.vocabSize,
    required this.requestedValueCount,
    required this.returnedValueCount,
    required this.epsilon,
    required this.meanSquare,
    required this.invRms,
    required this.min,
    required this.max,
    required this.mean,
    required this.l2Norm,
    required this.checksum,
    required this.weightShape,
    required this.values,
    this.error,
  });

  factory GgufMatVecResult.fromJson(Map<String, Object?> json) {
    return GgufMatVecResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      tokenId: _intValue(json['tokenId']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      normTensorName: _stringValue(json['normTensorName']),
      normTensorType: _stringValue(json['normTensorType']),
      weightTensorName: _stringValue(json['weightTensorName']),
      weightTensorType: _stringValue(json['weightTensorType']),
      inputLength: _intValue(json['inputLength']),
      outputLength: _intValue(json['outputLength']),
      vocabSize: _intValue(json['vocabSize']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedValueCount: _intValue(json['returnedValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      meanSquare: _doubleValue(json['meanSquare']),
      invRms: _doubleValue(json['invRms']),
      min: _doubleValue(json['min']),
      max: _doubleValue(json['max']),
      mean: _doubleValue(json['mean']),
      l2Norm: _doubleValue(json['l2Norm']),
      checksum: _doubleValue(json['checksum']),
      weightShape: _listValue(
        json['weightShape'],
      ).map((value) => _intValue(value)).toList(growable: false),
      values: _listValue(
        json['values'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final int tokenId;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String normTensorName;
  final String normTensorType;
  final String weightTensorName;
  final String weightTensorType;
  final int inputLength;
  final int outputLength;
  final int vocabSize;
  final int requestedValueCount;
  final int returnedValueCount;
  final double epsilon;
  final double meanSquare;
  final double invRms;
  final double min;
  final double max;
  final double mean;
  final double l2Norm;
  final double checksum;
  final List<int> weightShape;
  final List<double> values;
  final String? error;
}

class GgufRmsNormResult {
  const GgufRmsNormResult({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.tokenId,
    required this.embeddingTensorName,
    required this.embeddingTensorType,
    required this.normTensorName,
    required this.normTensorType,
    required this.embeddingLength,
    required this.vocabSize,
    required this.requestedValueCount,
    required this.returnedValueCount,
    required this.epsilon,
    required this.meanSquare,
    required this.invRms,
    required this.min,
    required this.max,
    required this.mean,
    required this.l2Norm,
    required this.checksum,
    required this.values,
    this.error,
  });

  factory GgufRmsNormResult.fromJson(Map<String, Object?> json) {
    return GgufRmsNormResult(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      tokenId: _intValue(json['tokenId']),
      embeddingTensorName: _stringValue(json['embeddingTensorName']),
      embeddingTensorType: _stringValue(json['embeddingTensorType']),
      normTensorName: _stringValue(json['normTensorName']),
      normTensorType: _stringValue(json['normTensorType']),
      embeddingLength: _intValue(json['embeddingLength']),
      vocabSize: _intValue(json['vocabSize']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedValueCount: _intValue(json['returnedValueCount']),
      epsilon: _doubleValue(json['epsilon']),
      meanSquare: _doubleValue(json['meanSquare']),
      invRms: _doubleValue(json['invRms']),
      min: _doubleValue(json['min']),
      max: _doubleValue(json['max']),
      mean: _doubleValue(json['mean']),
      l2Norm: _doubleValue(json['l2Norm']),
      checksum: _doubleValue(json['checksum']),
      values: _listValue(
        json['values'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final int tokenId;
  final String embeddingTensorName;
  final String embeddingTensorType;
  final String normTensorName;
  final String normTensorType;
  final int embeddingLength;
  final int vocabSize;
  final int requestedValueCount;
  final int returnedValueCount;
  final double epsilon;
  final double meanSquare;
  final double invRms;
  final double min;
  final double max;
  final double mean;
  final double l2Norm;
  final double checksum;
  final List<double> values;
  final String? error;
}

class GgufTokenEmbedding {
  const GgufTokenEmbedding({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.tensorName,
    required this.tensorType,
    required this.tensorTypeId,
    required this.tokenId,
    required this.embeddingLength,
    required this.vocabSize,
    required this.rowByteSize,
    required this.requestedValueCount,
    required this.returnedValueCount,
    required this.min,
    required this.max,
    required this.mean,
    required this.l2Norm,
    required this.checksum,
    required this.shape,
    required this.values,
    this.error,
  });

  factory GgufTokenEmbedding.fromJson(Map<String, Object?> json) {
    return GgufTokenEmbedding(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      tensorName: _stringValue(json['tensorName']),
      tensorType: _stringValue(json['tensorType']),
      tensorTypeId: _intValue(json['tensorTypeId']),
      tokenId: _intValue(json['tokenId']),
      embeddingLength: _intValue(json['embeddingLength']),
      vocabSize: _intValue(json['vocabSize']),
      rowByteSize: _intValue(json['rowByteSize']),
      requestedValueCount: _intValue(json['requestedValueCount']),
      returnedValueCount: _intValue(json['returnedValueCount']),
      min: _doubleValue(json['min']),
      max: _doubleValue(json['max']),
      mean: _doubleValue(json['mean']),
      l2Norm: _doubleValue(json['l2Norm']),
      checksum: _doubleValue(json['checksum']),
      shape: _listValue(
        json['shape'],
      ).map((value) => _intValue(value)).toList(growable: false),
      values: _listValue(
        json['values'],
      ).map((value) => _doubleValue(value)).toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final String tensorName;
  final String tensorType;
  final int tensorTypeId;
  final int tokenId;
  final int embeddingLength;
  final int vocabSize;
  final int rowByteSize;
  final int requestedValueCount;
  final int returnedValueCount;
  final double min;
  final double max;
  final double mean;
  final double l2Norm;
  final double checksum;
  final List<int> shape;
  final List<double> values;
  final String? error;
}

class GgufInspection {
  const GgufInspection({
    required this.ok,
    required this.path,
    required this.fileSizeBytes,
    required this.version,
    required this.tensorCount,
    required this.metadataCount,
    required this.alignment,
    required this.dataStartOffset,
    required this.parameterCount,
    required this.architecture,
    required this.name,
    required this.fileType,
    required this.tokenizerModel,
    required this.blockCount,
    required this.contextLength,
    required this.embeddingLength,
    required this.feedForwardLength,
    required this.headCount,
    required this.kvHeadCount,
    required this.ropeFreqBase,
    required this.tensorTypes,
    required this.metadata,
    required this.tensors,
    this.error,
  });

  factory GgufInspection.fromJson(Map<String, Object?> json) {
    return GgufInspection(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      fileSizeBytes: _intValue(json['fileSizeBytes']),
      version: _intValue(json['version']),
      tensorCount: _intValue(json['tensorCount']),
      metadataCount: _intValue(json['metadataCount']),
      alignment: _intValue(json['alignment']),
      dataStartOffset: _intValue(json['dataStartOffset']),
      parameterCount: _intValue(json['parameterCount']),
      architecture: _stringValue(json['architecture']),
      name: _stringValue(json['name']),
      fileType: _stringValue(json['fileType']),
      tokenizerModel: _stringValue(json['tokenizerModel']),
      blockCount: _intValue(json['blockCount']),
      contextLength: _intValue(json['contextLength']),
      embeddingLength: _intValue(json['embeddingLength']),
      feedForwardLength: _intValue(json['feedForwardLength']),
      headCount: _intValue(json['headCount']),
      kvHeadCount: _intValue(json['kvHeadCount']),
      ropeFreqBase: _stringValue(json['ropeFreqBase']),
      tensorTypes: _intMapValue(json['tensorTypes']),
      metadata: _stringMapValue(json['metadata']),
      tensors: _listValue(json['tensors'])
          .whereType<Map>()
          .map(
            (tensor) => GgufTensorInfo.fromJson(tensor.cast<String, Object?>()),
          )
          .toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final int fileSizeBytes;
  final int version;
  final int tensorCount;
  final int metadataCount;
  final int alignment;
  final int dataStartOffset;
  final int parameterCount;
  final String architecture;
  final String name;
  final String fileType;
  final String tokenizerModel;
  final int blockCount;
  final int contextLength;
  final int embeddingLength;
  final int feedForwardLength;
  final int headCount;
  final int kvHeadCount;
  final String ropeFreqBase;
  final Map<String, int> tensorTypes;
  final Map<String, String> metadata;
  final List<GgufTensorInfo> tensors;
  final String? error;
}

class GgufTensorInfo {
  const GgufTensorInfo({
    required this.name,
    required this.type,
    required this.typeId,
    required this.relativeOffset,
    required this.absoluteOffset,
    required this.elementCount,
    required this.hasKnownByteSize,
    required this.byteSize,
    required this.layerIndex,
    required this.shape,
  });

  factory GgufTensorInfo.fromJson(Map<String, Object?> json) {
    return GgufTensorInfo(
      name: _stringValue(json['name']),
      type: _stringValue(json['type']),
      typeId: _intValue(json['typeId']),
      relativeOffset: _intValue(json['relativeOffset']),
      absoluteOffset: _intValue(json['absoluteOffset']),
      elementCount: _intValue(json['elementCount']),
      hasKnownByteSize: _boolValue(json['hasKnownByteSize']),
      byteSize: _intValue(json['byteSize']),
      layerIndex: _intValue(json['layerIndex'], fallback: -1),
      shape: _listValue(
        json['shape'],
      ).map((value) => _intValue(value)).toList(growable: false),
    );
  }

  final String name;
  final String type;
  final int typeId;
  final int relativeOffset;
  final int absoluteOffset;
  final int elementCount;
  final bool hasKnownByteSize;
  final int byteSize;
  final int layerIndex;
  final List<int> shape;
}

class GgufLayerStreamingValidation {
  const GgufLayerStreamingValidation({
    required this.ok,
    required this.path,
    required this.architecture,
    required this.fileSizeBytes,
    required this.dataStartOffset,
    required this.blockCount,
    required this.layersDiscovered,
    required this.layersVisited,
    required this.tensorsVisited,
    required this.maxLayerSpanBytes,
    required this.totalLayerSpanBytes,
    required this.largestTensorBytes,
    required this.missingLayerCount,
    required this.missingLayers,
    required this.layers,
    this.error,
  });

  factory GgufLayerStreamingValidation.fromJson(Map<String, Object?> json) {
    return GgufLayerStreamingValidation(
      ok: _boolValue(json['ok']),
      path: _stringValue(json['path']),
      architecture: _stringValue(json['architecture']),
      fileSizeBytes: _intValue(json['fileSizeBytes']),
      dataStartOffset: _intValue(json['dataStartOffset']),
      blockCount: _intValue(json['blockCount']),
      layersDiscovered: _intValue(json['layersDiscovered']),
      layersVisited: _intValue(json['layersVisited']),
      tensorsVisited: _intValue(json['tensorsVisited']),
      maxLayerSpanBytes: _intValue(json['maxLayerSpanBytes']),
      totalLayerSpanBytes: _intValue(json['totalLayerSpanBytes']),
      largestTensorBytes: _intValue(json['largestTensorBytes']),
      missingLayerCount: _intValue(json['missingLayerCount']),
      missingLayers: _listValue(
        json['missingLayers'],
      ).map((value) => _intValue(value)).toList(growable: false),
      layers: _listValue(json['layers'])
          .whereType<Map>()
          .map(
            (layer) => GgufLayerSummary.fromJson(layer.cast<String, Object?>()),
          )
          .toList(growable: false),
      error: _nullableStringValue(json['error']),
    );
  }

  final bool ok;
  final String path;
  final String architecture;
  final int fileSizeBytes;
  final int dataStartOffset;
  final int blockCount;
  final int layersDiscovered;
  final int layersVisited;
  final int tensorsVisited;
  final int maxLayerSpanBytes;
  final int totalLayerSpanBytes;
  final int largestTensorBytes;
  final int missingLayerCount;
  final List<int> missingLayers;
  final List<GgufLayerSummary> layers;
  final String? error;
}

class GgufLayerSummary {
  const GgufLayerSummary({
    required this.index,
    required this.tensorCount,
    required this.spanStart,
    required this.spanBytes,
    required this.largestTensorBytes,
    required this.mapped,
  });

  factory GgufLayerSummary.fromJson(Map<String, Object?> json) {
    return GgufLayerSummary(
      index: _intValue(json['index']),
      tensorCount: _intValue(json['tensorCount']),
      spanStart: _intValue(json['spanStart']),
      spanBytes: _intValue(json['spanBytes']),
      largestTensorBytes: _intValue(json['largestTensorBytes']),
      mapped: _boolValue(json['mapped']),
    );
  }

  final int index;
  final int tensorCount;
  final int spanStart;
  final int spanBytes;
  final int largestTensorBytes;
  final bool mapped;
}

bool _boolValue(Object? value) {
  if (value is bool) return value;
  if (value is String) return value.toLowerCase() == 'true';
  return false;
}

int _intValue(Object? value, {int fallback = 0}) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  if (value is String) return int.tryParse(value) ?? fallback;
  return fallback;
}

double _doubleValue(Object? value, {double fallback = 0}) {
  if (value is double) return value;
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value) ?? fallback;
  return fallback;
}

String _stringValue(Object? value) {
  return value?.toString() ?? '';
}

String? _nullableStringValue(Object? value) {
  if (value == null) return null;
  final text = value.toString();
  return text.isEmpty ? null : text;
}

List<Object?> _listValue(Object? value) {
  if (value is List) return value.cast<Object?>();
  return const [];
}

List<int> _intListValue(Object? value) {
  return _listValue(
    value,
  ).map((item) => _intValue(item)).toList(growable: false);
}

List<double> _doubleListValue(Object? value) {
  return _listValue(
    value,
  ).map((item) => _doubleValue(item)).toList(growable: false);
}

Map<String, int> _intMapValue(Object? value) {
  if (value is! Map) return const {};
  return value.map(
    (key, mapValue) => MapEntry(key.toString(), _intValue(mapValue)),
  );
}

Map<String, String> _stringMapValue(Object? value) {
  if (value is! Map) return const {};
  return value.map(
    (key, mapValue) => MapEntry(key.toString(), mapValue.toString()),
  );
}
