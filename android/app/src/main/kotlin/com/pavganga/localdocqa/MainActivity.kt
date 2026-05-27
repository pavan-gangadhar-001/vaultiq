package com.pavganga.localdocqa

import android.os.Build
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val ggufRunner = NativeGgufRunner()
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            "local_doc_qa/device"
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "supportedAbis" -> result.success(Build.SUPPORTED_ABIS.toList())
                "externalModelCacheDir" -> {
                    val baseDir = externalMediaDirs.firstOrNull()
                    if (baseDir == null) {
                        result.success(null)
                    } else {
                        val modelDir = File(baseDir, "models")
                        if (!modelDir.exists()) {
                            modelDir.mkdirs()
                        }
                        result.success(modelDir.absolutePath)
                    }
                }
                "inspectGguf" -> {
                    val path = call.argument<String>("path")
                    if (path.isNullOrBlank()) {
                        result.error(
                            "invalid_argument",
                            "inspectGguf requires a non-empty path",
                            null
                        )
                    } else {
                        try {
                            result.success(ggufRunner.inspectGguf(path))
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_inspect_failed",
                                error.message ?: "GGUF inspection failed",
                                null
                            )
                        }
                    }
                }
                "validateGgufLayerStreaming" -> {
                    val path = call.argument<String>("path")
                    if (path.isNullOrBlank()) {
                        result.error(
                            "invalid_argument",
                            "validateGgufLayerStreaming requires a non-empty path",
                            null
                        )
                    } else {
                        try {
                            result.success(ggufRunner.validateLayerStreaming(path))
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_layer_streaming_failed",
                                error.message ?: "GGUF layer streaming validation failed",
                                null
                            )
                        }
                    }
                }
                "readGgufTokenEmbedding" -> {
                    val path = call.argument<String>("path")
                    val tokenId = call.argument<Int>("tokenId")
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (path.isNullOrBlank() || tokenId == null) {
                        result.error(
                            "invalid_argument",
                            "readGgufTokenEmbedding requires path and tokenId",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.readTokenEmbedding(
                                    path,
                                    tokenId,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_token_embedding_failed",
                                error.message ?: "GGUF token embedding read failed",
                                null
                            )
                        }
                    }
                }
                "rmsNormGgufTokenEmbedding" -> {
                    val path = call.argument<String>("path")
                    val tokenId = call.argument<Int>("tokenId")
                    val normTensorName = call.argument<String>("normTensorName")
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenId == null ||
                        normTensorName.isNullOrBlank()
                    ) {
                        result.error(
                            "invalid_argument",
                            "rmsNormGgufTokenEmbedding requires path, tokenId, and normTensorName",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.rmsNormTokenEmbedding(
                                    path,
                                    tokenId,
                                    normTensorName,
                                    epsilon.toFloat(),
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_rms_norm_failed",
                                error.message ?: "GGUF RMSNorm failed",
                                null
                            )
                        }
                    }
                }
                "matVecGgufRmsNormTokenEmbedding" -> {
                    val path = call.argument<String>("path")
                    val tokenId = call.argument<Int>("tokenId")
                    val normTensorName = call.argument<String>("normTensorName")
                    val weightTensorName = call.argument<String>("weightTensorName")
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenId == null ||
                        normTensorName.isNullOrBlank() ||
                        weightTensorName.isNullOrBlank()
                    ) {
                        result.error(
                            "invalid_argument",
                            "matVecGgufRmsNormTokenEmbedding requires path, tokenId, normTensorName, and weightTensorName",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.matVecRmsNormTokenEmbedding(
                                    path,
                                    tokenId,
                                    normTensorName,
                                    weightTensorName,
                                    epsilon.toFloat(),
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_mat_vec_failed",
                                error.message ?: "GGUF MatVec failed",
                                null
                            )
                        }
                    }
                }
                "ropeGgufMatVecRmsNormTokenEmbedding" -> {
                    val path = call.argument<String>("path")
                    val tokenId = call.argument<Int>("tokenId")
                    val normTensorName = call.argument<String>("normTensorName")
                    val weightTensorName = call.argument<String>("weightTensorName")
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val position = call.argument<Number>("position")?.toLong()
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenId == null ||
                        normTensorName.isNullOrBlank() ||
                        weightTensorName.isNullOrBlank() ||
                        position == null ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "ropeGgufMatVecRmsNormTokenEmbedding requires path, tokenId, normTensorName, weightTensorName, position, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.ropeMatVecRmsNormTokenEmbedding(
                                    path,
                                    tokenId,
                                    normTensorName,
                                    weightTensorName,
                                    epsilon.toFloat(),
                                    position,
                                    headDim,
                                    ropeTheta,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_rope_mat_vec_failed",
                                error.message ?: "GGUF RoPE MatVec failed",
                                null
                            )
                        }
                    }
                }
                "kvCacheGgufRmsNormTokenEmbeddings" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val normTensorName = call.argument<String>("normTensorName")
                    val keyWeightTensorName = call.argument<String>("keyWeightTensorName")
                    val valueWeightTensorName = call.argument<String>("valueWeightTensorName")
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val readPosition = call.argument<Number>("readPosition")?.toLong() ?: -1L
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        normTensorName.isNullOrBlank() ||
                        keyWeightTensorName.isNullOrBlank() ||
                        valueWeightTensorName.isNullOrBlank() ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "kvCacheGgufRmsNormTokenEmbeddings requires path, tokenIds, normTensorName, keyWeightTensorName, valueWeightTensorName, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.kvCacheRmsNormTokenEmbeddings(
                                    path,
                                    tokenIds.toIntArray(),
                                    normTensorName,
                                    keyWeightTensorName,
                                    valueWeightTensorName,
                                    epsilon.toFloat(),
                                    startPosition,
                                    readPosition,
                                    headDim,
                                    ropeTheta,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_kv_cache_failed",
                                error.message ?: "GGUF KV cache failed",
                                null
                            )
                        }
                    }
                }
                "attentionGgufSingleHead" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val queryTokenId = call.argument<Int>("queryTokenId")
                    val normTensorName = call.argument<String>("normTensorName")
                    val queryWeightTensorName = call.argument<String>("queryWeightTensorName")
                    val keyWeightTensorName = call.argument<String>("keyWeightTensorName")
                    val valueWeightTensorName = call.argument<String>("valueWeightTensorName")
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val queryPosition = call.argument<Number>("queryPosition")?.toLong()
                    val readPosition = call.argument<Int>("readPosition") ?: -1
                    val attentionLength = call.argument<Int>("attentionLength") ?: -1
                    val headDim = call.argument<Int>("headDim")
                    val headIndex = call.argument<Int>("headIndex") ?: 0
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        queryTokenId == null ||
                        normTensorName.isNullOrBlank() ||
                        queryWeightTensorName.isNullOrBlank() ||
                        keyWeightTensorName.isNullOrBlank() ||
                        valueWeightTensorName.isNullOrBlank() ||
                        queryPosition == null ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "attentionGgufSingleHead requires path, tokenIds, queryTokenId, normTensorName, queryWeightTensorName, keyWeightTensorName, valueWeightTensorName, queryPosition, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.attentionSingleHead(
                                    path,
                                    tokenIds.toIntArray(),
                                    queryTokenId,
                                    normTensorName,
                                    queryWeightTensorName,
                                    keyWeightTensorName,
                                    valueWeightTensorName,
                                    epsilon.toFloat(),
                                    startPosition,
                                    queryPosition,
                                    readPosition,
                                    attentionLength,
                                    headDim,
                                    headIndex,
                                    ropeTheta,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_attention_failed",
                                error.message ?: "GGUF attention failed",
                                null
                            )
                        }
                    }
                }
                "attentionGgufMultiHead" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val queryTokenId = call.argument<Int>("queryTokenId")
                    val normTensorName = call.argument<String>("normTensorName")
                    val queryWeightTensorName = call.argument<String>("queryWeightTensorName")
                    val keyWeightTensorName = call.argument<String>("keyWeightTensorName")
                    val valueWeightTensorName = call.argument<String>("valueWeightTensorName")
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val queryPosition = call.argument<Number>("queryPosition")?.toLong()
                    val readPosition = call.argument<Int>("readPosition") ?: -1
                    val attentionLength = call.argument<Int>("attentionLength") ?: -1
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        queryTokenId == null ||
                        normTensorName.isNullOrBlank() ||
                        queryWeightTensorName.isNullOrBlank() ||
                        keyWeightTensorName.isNullOrBlank() ||
                        valueWeightTensorName.isNullOrBlank() ||
                        queryPosition == null ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "attentionGgufMultiHead requires path, tokenIds, queryTokenId, normTensorName, queryWeightTensorName, keyWeightTensorName, valueWeightTensorName, queryPosition, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.attentionMultiHead(
                                    path,
                                    tokenIds.toIntArray(),
                                    queryTokenId,
                                    normTensorName,
                                    queryWeightTensorName,
                                    keyWeightTensorName,
                                    valueWeightTensorName,
                                    epsilon.toFloat(),
                                    startPosition,
                                    queryPosition,
                                    readPosition,
                                    attentionLength,
                                    headDim,
                                    ropeTheta,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_multi_head_attention_failed",
                                error.message ?: "GGUF multi-head attention failed",
                                null
                            )
                        }
                    }
                }
                "transformerLayerGguf" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val queryTokenId = call.argument<Int>("queryTokenId")
                    val attnNormTensorName = call.argument<String>("attnNormTensorName")
                    val queryWeightTensorName = call.argument<String>("queryWeightTensorName")
                    val keyWeightTensorName = call.argument<String>("keyWeightTensorName")
                    val valueWeightTensorName = call.argument<String>("valueWeightTensorName")
                    val outputWeightTensorName = call.argument<String>("outputWeightTensorName")
                    val ffnNormTensorName = call.argument<String>("ffnNormTensorName")
                    val gateWeightTensorName = call.argument<String>("gateWeightTensorName")
                    val upWeightTensorName = call.argument<String>("upWeightTensorName")
                    val downWeightTensorName = call.argument<String>("downWeightTensorName")
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val queryPosition = call.argument<Number>("queryPosition")?.toLong()
                    val readPosition = call.argument<Int>("readPosition") ?: -1
                    val attentionLength = call.argument<Int>("attentionLength") ?: -1
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        queryTokenId == null ||
                        attnNormTensorName.isNullOrBlank() ||
                        queryWeightTensorName.isNullOrBlank() ||
                        keyWeightTensorName.isNullOrBlank() ||
                        valueWeightTensorName.isNullOrBlank() ||
                        outputWeightTensorName.isNullOrBlank() ||
                        ffnNormTensorName.isNullOrBlank() ||
                        gateWeightTensorName.isNullOrBlank() ||
                        upWeightTensorName.isNullOrBlank() ||
                        downWeightTensorName.isNullOrBlank() ||
                        queryPosition == null ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "transformerLayerGguf requires path, tokenIds, queryTokenId, attention tensors, FFN tensors, queryPosition, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.transformerLayer(
                                    path,
                                    tokenIds.toIntArray(),
                                    queryTokenId,
                                    attnNormTensorName,
                                    queryWeightTensorName,
                                    keyWeightTensorName,
                                    valueWeightTensorName,
                                    outputWeightTensorName,
                                    ffnNormTensorName,
                                    gateWeightTensorName,
                                    upWeightTensorName,
                                    downWeightTensorName,
                                    epsilon.toFloat(),
                                    startPosition,
                                    queryPosition,
                                    readPosition,
                                    attentionLength,
                                    headDim,
                                    ropeTheta,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_transformer_layer_failed",
                                error.message ?: "GGUF transformer layer failed",
                                null
                            )
                        }
                    }
                }
                "transformerStackGguf" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val attnNormTensorSuffix =
                        call.argument<String>("attnNormTensorSuffix") ?: "attn_norm.weight"
                    val queryWeightTensorSuffix =
                        call.argument<String>("queryWeightTensorSuffix") ?: "attn_q.weight"
                    val keyWeightTensorSuffix =
                        call.argument<String>("keyWeightTensorSuffix") ?: "attn_k.weight"
                    val valueWeightTensorSuffix =
                        call.argument<String>("valueWeightTensorSuffix") ?: "attn_v.weight"
                    val outputWeightTensorSuffix =
                        call.argument<String>("outputWeightTensorSuffix") ?: "attn_o.weight"
                    val ffnNormTensorSuffix =
                        call.argument<String>("ffnNormTensorSuffix") ?: "ffn_norm.weight"
                    val gateWeightTensorSuffix =
                        call.argument<String>("gateWeightTensorSuffix") ?: "ffn_gate.weight"
                    val upWeightTensorSuffix =
                        call.argument<String>("upWeightTensorSuffix") ?: "ffn_up.weight"
                    val downWeightTensorSuffix =
                        call.argument<String>("downWeightTensorSuffix") ?: "ffn_down.weight"
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val readPosition = call.argument<Int>("readPosition") ?: -1
                    val layerCount = call.argument<Int>("layerCount") ?: -1
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        attnNormTensorSuffix.isBlank() ||
                        queryWeightTensorSuffix.isBlank() ||
                        keyWeightTensorSuffix.isBlank() ||
                        valueWeightTensorSuffix.isBlank() ||
                        outputWeightTensorSuffix.isBlank() ||
                        ffnNormTensorSuffix.isBlank() ||
                        gateWeightTensorSuffix.isBlank() ||
                        upWeightTensorSuffix.isBlank() ||
                        downWeightTensorSuffix.isBlank() ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "transformerStackGguf requires path, tokenIds, tensor suffixes, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.transformerStack(
                                    path,
                                    tokenIds.toIntArray(),
                                    attnNormTensorSuffix,
                                    queryWeightTensorSuffix,
                                    keyWeightTensorSuffix,
                                    valueWeightTensorSuffix,
                                    outputWeightTensorSuffix,
                                    ffnNormTensorSuffix,
                                    gateWeightTensorSuffix,
                                    upWeightTensorSuffix,
                                    downWeightTensorSuffix,
                                    epsilon.toFloat(),
                                    startPosition,
                                    readPosition,
                                    layerCount,
                                    headDim,
                                    ropeTheta,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_transformer_stack_failed",
                                error.message ?: "GGUF transformer stack failed",
                                null
                            )
                        }
                    }
                }
                "generateNextTokenGguf" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val attnNormTensorSuffix =
                        call.argument<String>("attnNormTensorSuffix") ?: "attn_norm.weight"
                    val queryWeightTensorSuffix =
                        call.argument<String>("queryWeightTensorSuffix") ?: "attn_q.weight"
                    val keyWeightTensorSuffix =
                        call.argument<String>("keyWeightTensorSuffix") ?: "attn_k.weight"
                    val valueWeightTensorSuffix =
                        call.argument<String>("valueWeightTensorSuffix") ?: "attn_v.weight"
                    val outputWeightTensorSuffix =
                        call.argument<String>("outputWeightTensorSuffix") ?: "attn_o.weight"
                    val ffnNormTensorSuffix =
                        call.argument<String>("ffnNormTensorSuffix") ?: "ffn_norm.weight"
                    val gateWeightTensorSuffix =
                        call.argument<String>("gateWeightTensorSuffix") ?: "ffn_gate.weight"
                    val upWeightTensorSuffix =
                        call.argument<String>("upWeightTensorSuffix") ?: "ffn_up.weight"
                    val downWeightTensorSuffix =
                        call.argument<String>("downWeightTensorSuffix") ?: "ffn_down.weight"
                    val finalNormTensorName =
                        call.argument<String>("finalNormTensorName") ?: "output_norm.weight"
                    val lmHeadTensorName =
                        call.argument<String>("lmHeadTensorName") ?: "output.weight"
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val readPosition = call.argument<Int>("readPosition") ?: -1
                    val layerCount = call.argument<Int>("layerCount") ?: -1
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val topK = call.argument<Int>("topK") ?: 5
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        attnNormTensorSuffix.isBlank() ||
                        queryWeightTensorSuffix.isBlank() ||
                        keyWeightTensorSuffix.isBlank() ||
                        valueWeightTensorSuffix.isBlank() ||
                        outputWeightTensorSuffix.isBlank() ||
                        ffnNormTensorSuffix.isBlank() ||
                        gateWeightTensorSuffix.isBlank() ||
                        upWeightTensorSuffix.isBlank() ||
                        downWeightTensorSuffix.isBlank() ||
                        finalNormTensorName.isBlank() ||
                        lmHeadTensorName.isBlank() ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "generateNextTokenGguf requires path, tokenIds, tensor suffixes, final norm, lm_head, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.generateNextToken(
                                    path,
                                    tokenIds.toIntArray(),
                                    attnNormTensorSuffix,
                                    queryWeightTensorSuffix,
                                    keyWeightTensorSuffix,
                                    valueWeightTensorSuffix,
                                    outputWeightTensorSuffix,
                                    ffnNormTensorSuffix,
                                    gateWeightTensorSuffix,
                                    upWeightTensorSuffix,
                                    downWeightTensorSuffix,
                                    finalNormTensorName,
                                    lmHeadTensorName,
                                    epsilon.toFloat(),
                                    startPosition,
                                    readPosition,
                                    layerCount,
                                    headDim,
                                    ropeTheta,
                                    topK,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_generate_next_token_failed",
                                error.message ?: "GGUF next-token generation failed",
                                null
                            )
                        }
                    }
                }
                "generateTokensGguf" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val attnNormTensorSuffix =
                        call.argument<String>("attnNormTensorSuffix") ?: "attn_norm.weight"
                    val queryWeightTensorSuffix =
                        call.argument<String>("queryWeightTensorSuffix") ?: "attn_q.weight"
                    val keyWeightTensorSuffix =
                        call.argument<String>("keyWeightTensorSuffix") ?: "attn_k.weight"
                    val valueWeightTensorSuffix =
                        call.argument<String>("valueWeightTensorSuffix") ?: "attn_v.weight"
                    val outputWeightTensorSuffix =
                        call.argument<String>("outputWeightTensorSuffix") ?: "attn_o.weight"
                    val ffnNormTensorSuffix =
                        call.argument<String>("ffnNormTensorSuffix") ?: "ffn_norm.weight"
                    val gateWeightTensorSuffix =
                        call.argument<String>("gateWeightTensorSuffix") ?: "ffn_gate.weight"
                    val upWeightTensorSuffix =
                        call.argument<String>("upWeightTensorSuffix") ?: "ffn_up.weight"
                    val downWeightTensorSuffix =
                        call.argument<String>("downWeightTensorSuffix") ?: "ffn_down.weight"
                    val finalNormTensorName =
                        call.argument<String>("finalNormTensorName") ?: "output_norm.weight"
                    val lmHeadTensorName =
                        call.argument<String>("lmHeadTensorName") ?: "output.weight"
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val readPosition = call.argument<Int>("readPosition") ?: -1
                    val layerCount = call.argument<Int>("layerCount") ?: -1
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxNewTokens = call.argument<Int>("maxNewTokens") ?: 1
                    val topK = call.argument<Int>("topK") ?: 5
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        attnNormTensorSuffix.isBlank() ||
                        queryWeightTensorSuffix.isBlank() ||
                        keyWeightTensorSuffix.isBlank() ||
                        valueWeightTensorSuffix.isBlank() ||
                        outputWeightTensorSuffix.isBlank() ||
                        ffnNormTensorSuffix.isBlank() ||
                        gateWeightTensorSuffix.isBlank() ||
                        upWeightTensorSuffix.isBlank() ||
                        downWeightTensorSuffix.isBlank() ||
                        finalNormTensorName.isBlank() ||
                        lmHeadTensorName.isBlank() ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "generateTokensGguf requires path, tokenIds, tensor suffixes, final norm, lm_head, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.generateTokens(
                                    path,
                                    tokenIds.toIntArray(),
                                    attnNormTensorSuffix,
                                    queryWeightTensorSuffix,
                                    keyWeightTensorSuffix,
                                    valueWeightTensorSuffix,
                                    outputWeightTensorSuffix,
                                    ffnNormTensorSuffix,
                                    gateWeightTensorSuffix,
                                    upWeightTensorSuffix,
                                    downWeightTensorSuffix,
                                    finalNormTensorName,
                                    lmHeadTensorName,
                                    epsilon.toFloat(),
                                    startPosition,
                                    readPosition,
                                    layerCount,
                                    headDim,
                                    ropeTheta,
                                    maxNewTokens,
                                    topK,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_generate_tokens_failed",
                                error.message ?: "GGUF greedy generation failed",
                                null
                            )
                        }
                    }
                }
                "createGgufSession" -> {
                    val path = call.argument<String>("path")
                    val tokenIdsArgument = call.argument<List<*>>("tokenIds")
                    val tokenIds = tokenIdsArgument
                        ?.mapNotNull { value -> (value as? Number)?.toInt() }
                    val attnNormTensorSuffix =
                        call.argument<String>("attnNormTensorSuffix") ?: "attn_norm.weight"
                    val queryWeightTensorSuffix =
                        call.argument<String>("queryWeightTensorSuffix") ?: "attn_q.weight"
                    val keyWeightTensorSuffix =
                        call.argument<String>("keyWeightTensorSuffix") ?: "attn_k.weight"
                    val valueWeightTensorSuffix =
                        call.argument<String>("valueWeightTensorSuffix") ?: "attn_v.weight"
                    val outputWeightTensorSuffix =
                        call.argument<String>("outputWeightTensorSuffix") ?: "attn_o.weight"
                    val ffnNormTensorSuffix =
                        call.argument<String>("ffnNormTensorSuffix") ?: "ffn_norm.weight"
                    val gateWeightTensorSuffix =
                        call.argument<String>("gateWeightTensorSuffix") ?: "ffn_gate.weight"
                    val upWeightTensorSuffix =
                        call.argument<String>("upWeightTensorSuffix") ?: "ffn_up.weight"
                    val downWeightTensorSuffix =
                        call.argument<String>("downWeightTensorSuffix") ?: "ffn_down.weight"
                    val finalNormTensorName =
                        call.argument<String>("finalNormTensorName") ?: "output_norm.weight"
                    val lmHeadTensorName =
                        call.argument<String>("lmHeadTensorName") ?: "output.weight"
                    val epsilon = call.argument<Double>("epsilon") ?: 0.000001
                    val startPosition = call.argument<Number>("startPosition")?.toLong() ?: 0L
                    val readPosition = call.argument<Int>("readPosition") ?: -1
                    val layerCount = call.argument<Int>("layerCount") ?: -1
                    val headDim = call.argument<Int>("headDim")
                    val ropeTheta = call.argument<Number>("ropeTheta")?.toDouble() ?: 0.0
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (
                        path.isNullOrBlank() ||
                        tokenIdsArgument == null ||
                        tokenIds == null ||
                        tokenIds.isEmpty() ||
                        tokenIds.size != tokenIdsArgument.size ||
                        attnNormTensorSuffix.isBlank() ||
                        queryWeightTensorSuffix.isBlank() ||
                        keyWeightTensorSuffix.isBlank() ||
                        valueWeightTensorSuffix.isBlank() ||
                        outputWeightTensorSuffix.isBlank() ||
                        ffnNormTensorSuffix.isBlank() ||
                        gateWeightTensorSuffix.isBlank() ||
                        upWeightTensorSuffix.isBlank() ||
                        downWeightTensorSuffix.isBlank() ||
                        finalNormTensorName.isBlank() ||
                        lmHeadTensorName.isBlank() ||
                        headDim == null
                    ) {
                        result.error(
                            "invalid_argument",
                            "createGgufSession requires path, tokenIds, tensor suffixes, final norm, lm_head, and headDim",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.createSession(
                                    path,
                                    tokenIds.toIntArray(),
                                    attnNormTensorSuffix,
                                    queryWeightTensorSuffix,
                                    keyWeightTensorSuffix,
                                    valueWeightTensorSuffix,
                                    outputWeightTensorSuffix,
                                    ffnNormTensorSuffix,
                                    gateWeightTensorSuffix,
                                    upWeightTensorSuffix,
                                    downWeightTensorSuffix,
                                    finalNormTensorName,
                                    lmHeadTensorName,
                                    epsilon.toFloat(),
                                    startPosition,
                                    readPosition,
                                    layerCount,
                                    headDim,
                                    ropeTheta,
                                    maxValues
                                )
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_create_session_failed",
                                error.message ?: "GGUF session creation failed",
                                null
                            )
                        }
                    }
                }
                "decodeGgufSession" -> {
                    val sessionId = call.argument<Number>("sessionId")?.toLong()
                    val topK = call.argument<Int>("topK") ?: 5
                    val maxValues = call.argument<Int>("maxValues") ?: 16
                    if (sessionId == null || sessionId <= 0L) {
                        result.error(
                            "invalid_argument",
                            "decodeGgufSession requires a positive sessionId",
                            null
                        )
                    } else {
                        try {
                            result.success(
                                ggufRunner.decodeSession(sessionId, topK, maxValues)
                            )
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_decode_session_failed",
                                error.message ?: "GGUF session decode failed",
                                null
                            )
                        }
                    }
                }
                "closeGgufSession" -> {
                    val sessionId = call.argument<Number>("sessionId")?.toLong()
                    if (sessionId == null || sessionId <= 0L) {
                        result.error(
                            "invalid_argument",
                            "closeGgufSession requires a positive sessionId",
                            null
                        )
                    } else {
                        try {
                            result.success(ggufRunner.closeSession(sessionId))
                        } catch (error: Throwable) {
                            result.error(
                                "gguf_close_session_failed",
                                error.message ?: "GGUF session close failed",
                                null
                            )
                        }
                    }
                }
                else -> result.notImplemented()
            }
        }
    }
}
