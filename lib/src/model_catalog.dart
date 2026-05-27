import 'package:flutter_gemma/flutter_gemma.dart';

const int _decimalMb = 1000 * 1000;
const int _decimalGb = 1000 * _decimalMb;

enum GgufModelRole { chatLlm, embedding }

class GgufModelFile {
  const GgufModelFile({
    required this.label,
    required this.repository,
    required this.filename,
    required this.displaySize,
    required this.sizeBytes,
    required this.role,
    required this.license,
    this.parameterCount,
    this.layerCount,
    this.queryHeadCount,
    this.kvHeadCount,
    this.nativeContextTokens,
    this.minEmbeddingDimensions,
    this.maxEmbeddingDimensions,
    this.languageSupport,
    this.supportsThinking = false,
  });

  final String label;
  final String repository;
  final String filename;
  final String displaySize;
  final int sizeBytes;
  final GgufModelRole role;
  final String license;
  final String? parameterCount;
  final int? layerCount;
  final int? queryHeadCount;
  final int? kvHeadCount;
  final int? nativeContextTokens;
  final int? minEmbeddingDimensions;
  final int? maxEmbeddingDimensions;
  final String? languageSupport;
  final bool supportsThinking;

  String get id => '$repository/$filename';

  String get url => 'https://huggingface.co/$repository/resolve/main/$filename';
}

class GgufModelBundle {
  const GgufModelBundle({
    required this.label,
    required this.llm,
    this.embedding,
    required this.description,
  });

  final String label;
  final GgufModelFile llm;
  final GgufModelFile? embedding;
  final String description;

  int get totalSizeBytes => llm.sizeBytes + (embedding?.sizeBytes ?? 0);

  double get totalSizeGb => totalSizeBytes / _decimalGb;

  bool fitsStorageBudgetBytes(int storageBudgetBytes) {
    return totalSizeBytes <= storageBudgetBytes;
  }

  bool fitsStorageBudgetGb(double storageBudgetGb) {
    return totalSizeBytes <= storageBudgetGb * _decimalGb;
  }
}

class RecommendedGgufModels {
  const RecommendedGgufModels._();

  static const storageBudgetBytes = 6 * _decimalGb;

  static const qwen3EightBQ4KM = GgufModelFile(
    label: 'Qwen3-8B Q4_K_M',
    repository: 'Qwen/Qwen3-8B-GGUF',
    filename: 'Qwen3-8B-Q4_K_M.gguf',
    displaySize: '5.03 GB',
    sizeBytes: 5030 * _decimalMb,
    role: GgufModelRole.chatLlm,
    license: 'Apache-2.0',
    parameterCount: '8.2B',
    layerCount: 36,
    queryHeadCount: 32,
    kvHeadCount: 8,
    nativeContextTokens: 32768,
    supportsThinking: true,
  );

  static const qwen3EightBQ5KM = GgufModelFile(
    label: 'Qwen3-8B Q5_K_M',
    repository: 'Qwen/Qwen3-8B-GGUF',
    filename: 'Qwen3-8B-Q5_K_M.gguf',
    displaySize: '5.85 GB',
    sizeBytes: 5850 * _decimalMb,
    role: GgufModelRole.chatLlm,
    license: 'Apache-2.0',
    parameterCount: '8.2B',
    layerCount: 36,
    queryHeadCount: 32,
    kvHeadCount: 8,
    nativeContextTokens: 32768,
    supportsThinking: true,
  );

  static const qwen3FourBQ8 = GgufModelFile(
    label: 'Qwen3-4B Q8_0',
    repository: 'Qwen/Qwen3-4B-GGUF',
    filename: 'Qwen3-4B-Q8_0.gguf',
    displaySize: '4.28 GB',
    sizeBytes: 4280 * _decimalMb,
    role: GgufModelRole.chatLlm,
    license: 'Apache-2.0',
    parameterCount: '4.0B',
    nativeContextTokens: 32768,
    supportsThinking: true,
  );

  static const qwen3Embedding06BQ8 = GgufModelFile(
    label: 'Qwen3-Embedding-0.6B Q8_0',
    repository: 'Qwen/Qwen3-Embedding-0.6B-GGUF',
    filename: 'Qwen3-Embedding-0.6B-Q8_0.gguf',
    displaySize: '639 MB',
    sizeBytes: 639 * _decimalMb,
    role: GgufModelRole.embedding,
    license: 'Apache-2.0',
    parameterCount: '0.6B',
    nativeContextTokens: 32768,
    minEmbeddingDimensions: 32,
    maxEmbeddingDimensions: 1024,
    languageSupport: '100+ languages',
  );

  static const primaryBuild = GgufModelBundle(
    label: 'Primary 6 GB target',
    llm: qwen3EightBQ4KM,
    embedding: qwen3Embedding06BQ8,
    description:
        'Best fit when total model storage is capped at 6 GB and both chat '
        'generation and local semantic retrieval are required.',
  );

  static const fallbackDebugBuild = GgufModelBundle(
    label: 'Fallback/debug target',
    llm: qwen3FourBQ8,
    embedding: qwen3Embedding06BQ8,
    description:
        'Lower-risk validation bundle for Android storage and RAM bring-up '
        'before switching to the 8B Q4_K_M target.',
  );

  static const llmOnlyBuild = GgufModelBundle(
    label: 'LLM-only target',
    llm: qwen3EightBQ5KM,
    description:
        'Higher-quality single-file chat model under 6 GB when no separate '
        'embedding model is installed.',
  );

  static const allBundles = [primaryBuild, fallbackDebugBuild, llmOnlyBuild];
}

enum DownloadableModel {
  qwen25OnePointFiveB(
    label: 'Local answer engine',
    size: '1.6 GB',
    description: 'Required for private answers that run on this device.',
    url:
        'https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/main/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.task',
    modelType: ModelType.qwen,
    fileType: ModelFileType.task,
    preferredBackend: PreferredBackend.cpu,
    maxTokens: 4096,
    temperature: 0.1,
    topK: 24,
    topP: 0.8,
    reservedOutputTokens: 640,
    maxPromptHits: 6,
    maxContextChars: 4800,
    maxExcerptChars: 900,
    maxQuestionChars: 900,
  );

  const DownloadableModel({
    required this.label,
    required this.size,
    required this.description,
    required this.url,
    required this.modelType,
    required this.fileType,
    required this.preferredBackend,
    required this.maxTokens,
    required this.temperature,
    required this.topK,
    required this.topP,
    this.reservedOutputTokens = 512,
    this.maxPromptHits = 6,
    this.maxContextChars = 6000,
    this.maxExcerptChars = 900,
    this.maxQuestionChars = 900,
  });

  final String label;
  final String size;
  final String description;
  final String url;
  final ModelType modelType;
  final ModelFileType fileType;
  final PreferredBackend preferredBackend;
  final int maxTokens;
  final double temperature;
  final int topK;
  final double topP;
  final int reservedOutputTokens;
  final int maxPromptHits;
  final int maxContextChars;
  final int maxExcerptChars;
  final int maxQuestionChars;

  bool get useForegroundDownload => true;
  String get filename => Uri.parse(url).pathSegments.last;

  LocalInferenceConfig get inferenceConfig {
    return LocalInferenceConfig(
      preferredBackend: preferredBackend,
      maxTokens: maxTokens,
      temperature: temperature,
      topK: topK,
      topP: topP,
      reservedOutputTokens: reservedOutputTokens,
      maxPromptHits: maxPromptHits,
      maxContextChars: maxContextChars,
      maxExcerptChars: maxExcerptChars,
      maxQuestionChars: maxQuestionChars,
    );
  }
}

class LocalInferenceConfig {
  const LocalInferenceConfig({
    this.preferredBackend = PreferredBackend.cpu,
    this.maxTokens = 4096,
    this.temperature = 0.2,
    this.topK = 40,
    this.topP = 0.9,
    this.reservedOutputTokens = 512,
    this.maxPromptHits = 6,
    this.maxContextChars = 6000,
    this.maxExcerptChars = 900,
    this.maxQuestionChars = 900,
  }) : assert(maxTokens > 0),
       assert(reservedOutputTokens >= 0),
       assert(maxPromptHits > 0),
       assert(maxContextChars > 0),
       assert(maxExcerptChars > 0),
       assert(maxQuestionChars > 0);

  final PreferredBackend preferredBackend;
  final int maxTokens;
  final double temperature;
  final int topK;
  final double topP;
  final int reservedOutputTokens;
  final int maxPromptHits;
  final int maxContextChars;
  final int maxExcerptChars;
  final int maxQuestionChars;

  int get promptTokenTarget {
    final hardLimit = hardPromptTokenLimit;
    final target = maxTokens - reservedOutputTokens;
    final minimum = hardLimit < 256 ? hardLimit : 256;
    if (target < minimum) return minimum;
    if (target > hardLimit) return hardLimit;
    return target;
  }

  int get hardPromptTokenLimit {
    if (maxTokens <= 1) return 1;
    final guard = maxTokens > 128 ? 64 : 1;
    final limit = maxTokens - guard;
    return limit < 1 ? 1 : limit;
  }
}

enum EmbeddingRuntime { flutterGemma, dartHash }

enum DownloadableEmbeddingModel {
  gecko256(
    label: 'Semantic search engine',
    size: '114 MB',
    description: 'Required for semantic matching across local files.',
    url:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_256_quant.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/sentencepiece.model',
    iosTokenizerUrl:
        'https://github.com/DenisovAV/flutter_gemma/releases/download/v0.12.5/gecko_tokenizer.json',
    filename: 'Gecko_256_quant.tflite',
    dimension: 768,
    maxSequenceLength: 256,
    maxInputChars: 220,
  ),
  debugHashing(
    label: 'Emulator semantic search',
    size: 'Built in',
    description:
        'Deterministic Dart embeddings for emulator and CI retrieval tests.',
    url: '',
    tokenizerUrl: '',
    iosTokenizerUrl: '',
    filename: 'debug_hashing_v1',
    dimension: 384,
    maxSequenceLength: 2048,
    maxInputChars: 1400,
    runtime: EmbeddingRuntime.dartHash,
  );

  const DownloadableEmbeddingModel({
    required this.label,
    required this.size,
    required this.description,
    required this.url,
    required this.tokenizerUrl,
    required this.iosTokenizerUrl,
    required this.filename,
    required this.dimension,
    required this.maxSequenceLength,
    required this.maxInputChars,
    this.runtime = EmbeddingRuntime.flutterGemma,
  });

  final String label;
  final String size;
  final String description;
  final String url;
  final String tokenizerUrl;
  final String iosTokenizerUrl;
  final String filename;
  final int dimension;
  final int maxSequenceLength;
  final int maxInputChars;
  final EmbeddingRuntime runtime;

  String get id => filename;
  bool get isBuiltIn => runtime == EmbeddingRuntime.dartHash;
  bool get requiresNetworkInstall => runtime == EmbeddingRuntime.flutterGemma;
}
