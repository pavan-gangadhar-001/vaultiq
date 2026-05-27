import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/model_catalog.dart';

void main() {
  test(
    'primary GGUF bundle uses the selected Qwen3 8B plus embedding setup',
    () {
      final bundle = RecommendedGgufModels.primaryBuild;

      expect(bundle.llm.filename, 'Qwen3-8B-Q4_K_M.gguf');
      expect(bundle.llm.repository, 'Qwen/Qwen3-8B-GGUF');
      expect(bundle.llm.parameterCount, '8.2B');
      expect(bundle.llm.layerCount, 36);
      expect(bundle.llm.queryHeadCount, 32);
      expect(bundle.llm.kvHeadCount, 8);
      expect(bundle.llm.nativeContextTokens, 32768);
      expect(bundle.llm.supportsThinking, isTrue);

      expect(bundle.embedding?.filename, 'Qwen3-Embedding-0.6B-Q8_0.gguf');
      expect(bundle.embedding?.maxEmbeddingDimensions, 1024);
      expect(bundle.embedding?.minEmbeddingDimensions, 32);
      expect(bundle.embedding?.nativeContextTokens, 32768);
    },
  );

  test('primary GGUF bundle fits the 6 GB storage budget', () {
    final bundle = RecommendedGgufModels.primaryBuild;

    expect(bundle.totalSizeBytes, 5669 * 1000 * 1000);
    expect(
      bundle.fitsStorageBudgetBytes(RecommendedGgufModels.storageBudgetBytes),
      isTrue,
    );
    expect(bundle.fitsStorageBudgetGb(6), isTrue);
  });

  test('Qwen3 8B Q5_K_M is only a 6 GB option without embeddings', () {
    final llmOnly = RecommendedGgufModels.llmOnlyBuild;
    final llmWithEmbeddingBytes =
        llmOnly.llm.sizeBytes +
        RecommendedGgufModels.qwen3Embedding06BQ8.sizeBytes;

    expect(llmOnly.llm.filename, 'Qwen3-8B-Q5_K_M.gguf');
    expect(llmOnly.fitsStorageBudgetGb(6), isTrue);
    expect(
      llmWithEmbeddingBytes,
      greaterThan(RecommendedGgufModels.storageBudgetBytes),
    );
  });

  test('fallback GGUF bundle keeps extra Android bring-up margin', () {
    final bundle = RecommendedGgufModels.fallbackDebugBuild;

    expect(bundle.llm.filename, 'Qwen3-4B-Q8_0.gguf');
    expect(bundle.totalSizeBytes, 4919 * 1000 * 1000);
    expect(bundle.fitsStorageBudgetGb(5), isTrue);
  });

  test('debug hashing embedder is built in for emulator testing', () {
    const model = DownloadableEmbeddingModel.debugHashing;

    expect(model.runtime, EmbeddingRuntime.dartHash);
    expect(model.isBuiltIn, isTrue);
    expect(model.requiresNetworkInstall, isFalse);
    expect(model.dimension, 384);
    expect(
      model.maxInputChars,
      greaterThan(DownloadableEmbeddingModel.gecko256.maxInputChars),
    );
  });
}
