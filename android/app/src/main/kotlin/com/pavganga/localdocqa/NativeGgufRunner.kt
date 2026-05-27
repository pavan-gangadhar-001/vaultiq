package com.pavganga.localdocqa

class NativeGgufRunner {
    companion object {
        init {
            System.loadLibrary("vaultiq_gguf_runner")
        }
    }

    fun inspectGguf(path: String): String {
        return inspectGgufNative(path)
    }

    fun validateLayerStreaming(path: String): String {
        return validateLayerStreamingNative(path)
    }

    fun readTokenEmbedding(path: String, tokenId: Int, maxValues: Int): String {
        return readTokenEmbeddingNative(path, tokenId, maxValues)
    }

    fun rmsNormTokenEmbedding(
        path: String,
        tokenId: Int,
        normTensorName: String,
        epsilon: Float,
        maxValues: Int
    ): String {
        return rmsNormTokenEmbeddingNative(
            path,
            tokenId,
            normTensorName,
            epsilon,
            maxValues
        )
    }

    fun matVecRmsNormTokenEmbedding(
        path: String,
        tokenId: Int,
        normTensorName: String,
        weightTensorName: String,
        epsilon: Float,
        maxValues: Int
    ): String {
        return matVecRmsNormTokenEmbeddingNative(
            path,
            tokenId,
            normTensorName,
            weightTensorName,
            epsilon,
            maxValues
        )
    }

    fun ropeMatVecRmsNormTokenEmbedding(
        path: String,
        tokenId: Int,
        normTensorName: String,
        weightTensorName: String,
        epsilon: Float,
        position: Long,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String {
        return ropeMatVecRmsNormTokenEmbeddingNative(
            path,
            tokenId,
            normTensorName,
            weightTensorName,
            epsilon,
            position,
            headDim,
            ropeTheta,
            maxValues
        )
    }

    fun kvCacheRmsNormTokenEmbeddings(
        path: String,
        tokenIds: IntArray,
        normTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Long,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String {
        return kvCacheRmsNormTokenEmbeddingsNative(
            path,
            tokenIds,
            normTensorName,
            keyWeightTensorName,
            valueWeightTensorName,
            epsilon,
            startPosition,
            readPosition,
            headDim,
            ropeTheta,
            maxValues
        )
    }

    fun attentionSingleHead(
        path: String,
        tokenIds: IntArray,
        queryTokenId: Int,
        normTensorName: String,
        queryWeightTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        queryPosition: Long,
        readPosition: Int,
        attentionLength: Int,
        headDim: Int,
        headIndex: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String {
        return attentionSingleHeadNative(
            path,
            tokenIds,
            queryTokenId,
            normTensorName,
            queryWeightTensorName,
            keyWeightTensorName,
            valueWeightTensorName,
            epsilon,
            startPosition,
            queryPosition,
            readPosition,
            attentionLength,
            headDim,
            headIndex,
            ropeTheta,
            maxValues
        )
    }

    fun attentionMultiHead(
        path: String,
        tokenIds: IntArray,
        queryTokenId: Int,
        normTensorName: String,
        queryWeightTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        queryPosition: Long,
        readPosition: Int,
        attentionLength: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String {
        return attentionMultiHeadNative(
            path,
            tokenIds,
            queryTokenId,
            normTensorName,
            queryWeightTensorName,
            keyWeightTensorName,
            valueWeightTensorName,
            epsilon,
            startPosition,
            queryPosition,
            readPosition,
            attentionLength,
            headDim,
            ropeTheta,
            maxValues
        )
    }

    fun transformerLayer(
        path: String,
        tokenIds: IntArray,
        queryTokenId: Int,
        attnNormTensorName: String,
        queryWeightTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        outputWeightTensorName: String,
        ffnNormTensorName: String,
        gateWeightTensorName: String,
        upWeightTensorName: String,
        downWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        queryPosition: Long,
        readPosition: Int,
        attentionLength: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String {
        return transformerLayerNative(
            path,
            tokenIds,
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
            epsilon,
            startPosition,
            queryPosition,
            readPosition,
            attentionLength,
            headDim,
            ropeTheta,
            maxValues
        )
    }

    fun transformerStack(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String {
        return transformerStackNative(
            path,
            tokenIds,
            attnNormTensorSuffix,
            queryWeightTensorSuffix,
            keyWeightTensorSuffix,
            valueWeightTensorSuffix,
            outputWeightTensorSuffix,
            ffnNormTensorSuffix,
            gateWeightTensorSuffix,
            upWeightTensorSuffix,
            downWeightTensorSuffix,
            epsilon,
            startPosition,
            readPosition,
            layerCount,
            headDim,
            ropeTheta,
            maxValues
        )
    }

    fun generateNextToken(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        finalNormTensorName: String,
        lmHeadTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        topK: Int,
        maxValues: Int
    ): String {
        return generateNextTokenNative(
            path,
            tokenIds,
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
            epsilon,
            startPosition,
            readPosition,
            layerCount,
            headDim,
            ropeTheta,
            topK,
            maxValues
        )
    }

    fun generateTokens(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        finalNormTensorName: String,
        lmHeadTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        maxNewTokens: Int,
        topK: Int,
        maxValues: Int
    ): String {
        return generateTokensNative(
            path,
            tokenIds,
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
            epsilon,
            startPosition,
            readPosition,
            layerCount,
            headDim,
            ropeTheta,
            maxNewTokens,
            topK,
            maxValues
        )
    }

    fun createSession(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        finalNormTensorName: String,
        lmHeadTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String {
        return createSessionNative(
            path,
            tokenIds,
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
            epsilon,
            startPosition,
            readPosition,
            layerCount,
            headDim,
            ropeTheta,
            maxValues
        )
    }

    fun decodeSession(sessionId: Long, topK: Int, maxValues: Int): String {
        return decodeSessionNative(sessionId, topK, maxValues)
    }

    fun closeSession(sessionId: Long): String {
        return closeSessionNative(sessionId)
    }

    private external fun inspectGgufNative(path: String): String
    private external fun validateLayerStreamingNative(path: String): String
    private external fun readTokenEmbeddingNative(
        path: String,
        tokenId: Int,
        maxValues: Int
    ): String
    private external fun rmsNormTokenEmbeddingNative(
        path: String,
        tokenId: Int,
        normTensorName: String,
        epsilon: Float,
        maxValues: Int
    ): String
    private external fun matVecRmsNormTokenEmbeddingNative(
        path: String,
        tokenId: Int,
        normTensorName: String,
        weightTensorName: String,
        epsilon: Float,
        maxValues: Int
    ): String
    private external fun ropeMatVecRmsNormTokenEmbeddingNative(
        path: String,
        tokenId: Int,
        normTensorName: String,
        weightTensorName: String,
        epsilon: Float,
        position: Long,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String
    private external fun kvCacheRmsNormTokenEmbeddingsNative(
        path: String,
        tokenIds: IntArray,
        normTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Long,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String
    private external fun attentionSingleHeadNative(
        path: String,
        tokenIds: IntArray,
        queryTokenId: Int,
        normTensorName: String,
        queryWeightTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        queryPosition: Long,
        readPosition: Int,
        attentionLength: Int,
        headDim: Int,
        headIndex: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String
    private external fun attentionMultiHeadNative(
        path: String,
        tokenIds: IntArray,
        queryTokenId: Int,
        normTensorName: String,
        queryWeightTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        queryPosition: Long,
        readPosition: Int,
        attentionLength: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String
    private external fun transformerLayerNative(
        path: String,
        tokenIds: IntArray,
        queryTokenId: Int,
        attnNormTensorName: String,
        queryWeightTensorName: String,
        keyWeightTensorName: String,
        valueWeightTensorName: String,
        outputWeightTensorName: String,
        ffnNormTensorName: String,
        gateWeightTensorName: String,
        upWeightTensorName: String,
        downWeightTensorName: String,
        epsilon: Float,
        startPosition: Long,
        queryPosition: Long,
        readPosition: Int,
        attentionLength: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String
    private external fun transformerStackNative(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String
    private external fun generateNextTokenNative(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        finalNormTensorName: String,
        lmHeadTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        topK: Int,
        maxValues: Int
    ): String
    private external fun generateTokensNative(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        finalNormTensorName: String,
        lmHeadTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        maxNewTokens: Int,
        topK: Int,
        maxValues: Int
    ): String
    private external fun createSessionNative(
        path: String,
        tokenIds: IntArray,
        attnNormTensorSuffix: String,
        queryWeightTensorSuffix: String,
        keyWeightTensorSuffix: String,
        valueWeightTensorSuffix: String,
        outputWeightTensorSuffix: String,
        ffnNormTensorSuffix: String,
        gateWeightTensorSuffix: String,
        upWeightTensorSuffix: String,
        downWeightTensorSuffix: String,
        finalNormTensorName: String,
        lmHeadTensorName: String,
        epsilon: Float,
        startPosition: Long,
        readPosition: Int,
        layerCount: Int,
        headDim: Int,
        ropeTheta: Double,
        maxValues: Int
    ): String
    private external fun decodeSessionNative(
        sessionId: Long,
        topK: Int,
        maxValues: Int
    ): String
    private external fun closeSessionNative(sessionId: Long): String
}
