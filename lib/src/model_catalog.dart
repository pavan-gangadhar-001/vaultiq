import 'package:flutter_gemma/flutter_gemma.dart';

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

  String get id => filename;
}
