import 'dart:async';
import 'dart:io';

import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

import '../model_catalog.dart';
import '../models.dart';
import 'app_logger.dart';

class LocalAiService {
  static const MethodChannel _deviceChannel = MethodChannel(
    'local_doc_qa/device',
  );
  static const int _maxSafeEmbeddingInputChars = 220;
  static const int _minEmbeddingRetryInputChars = 64;
  static const Set<String> _queryStopWords = {
    'about',
    'answer',
    'case',
    'company',
    'date',
    'does',
    'file',
    'from',
    'give',
    'have',
    'letter',
    'local',
    'location',
    'offer',
    'please',
    'provide',
    'tell',
    'that',
    'the',
    'this',
    'what',
    'when',
    'where',
    'which',
    'with',
  };

  InferenceModel? _model;
  EmbeddingModel? _embedder;
  Future<void> _nativeOperation = Future.value();
  int _nativeOperationSequence = 0;
  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) {
      AppLogger.info('ai.initialize.skip_already_initialized');
      return;
    }
    final stopwatch = Stopwatch()..start();
    AppLogger.info('ai.initialize.start');
    await FlutterGemma.initialize();
    _initialized = true;
    AppLogger.info('ai.initialize.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'hasActiveModel': hasActiveModel,
      'hasActiveEmbedder': hasActiveEmbedder,
    });
  }

  bool get hasActiveModel {
    if (!_initialized) return false;
    return FlutterGemma.hasActiveModel();
  }

  bool get hasActiveEmbedder {
    if (!_initialized) return false;
    return FlutterGemma.hasActiveEmbedder();
  }

  Future<bool> isModelInstalled(DownloadableModel model) async {
    await initialize();
    final installed = await FlutterGemma.isModelInstalled(model.filename);
    AppLogger.info('ai.model.installed.checked', {
      'filename': model.filename,
      'installed': installed,
      'hasActiveModel': hasActiveModel,
    });
    return installed;
  }

  Future<List<String>> supportedAbis() async {
    if (!Platform.isAndroid) {
      AppLogger.info('ai.supported_abis.non_android');
      return const [];
    }
    try {
      final androidInfo = await DeviceInfoPlugin().androidInfo;
      if (androidInfo.supportedAbis.isNotEmpty) {
        AppLogger.info('ai.supported_abis.device_info', {
          'abis': androidInfo.supportedAbis,
        });
        return androidInfo.supportedAbis;
      }
    } catch (e, st) {
      AppLogger.error('ai.supported_abis.device_info_error', e, st);
      // Fall through to the app-local channel for environments where the
      // device_info_plus channel is unavailable.
    }

    try {
      final abis =
          await _deviceChannel.invokeListMethod<String>('supportedAbis') ??
          const [];
      AppLogger.info('ai.supported_abis.method_channel', {'abis': abis});
      return abis;
    } on MissingPluginException {
      AppLogger.warn('ai.supported_abis.missing_plugin');
      return const [];
    }
  }

  Future<bool> supportsLocalEmbedder() async {
    if (!Platform.isAndroid) {
      AppLogger.info('ai.embedder_support.non_android', {'supported': true});
      return true;
    }
    final abis = await supportedAbis();
    final supported = abis.isEmpty || abis.first == 'arm64-v8a';
    AppLogger.info('ai.embedder_support.checked', {
      'abis': abis,
      'supported': supported,
    });
    return supported;
  }

  Future<void> installModelFromFile({
    required String path,
    required ModelType modelType,
    required ModelFileType fileType,
  }) async {
    final stopwatch = Stopwatch()..start();
    AppLogger.info('ai.install_model_file.start', {
      'path': path,
      'modelType': modelType.name,
      'fileType': fileType.name,
    });
    await initialize();
    await _closeEmbedder();
    await FlutterGemma.installModel(
      modelType: modelType,
      fileType: fileType,
    ).fromFile(path).install();
    _model = null;
    AppLogger.info('ai.install_model_file.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'hasActiveModel': hasActiveModel,
    });
  }

  Future<void> installModelFromNetwork({
    required DownloadableModel model,
    required void Function(int progress) onProgress,
  }) async {
    final stopwatch = Stopwatch()..start();
    AppLogger.info('ai.install_model_network.start', {
      'label': model.label,
      'size': model.size,
      'modelType': model.modelType.name,
      'fileType': model.fileType.name,
      'foreground': model.useForegroundDownload,
    });
    await initialize();
    await _closeEmbedder();
    await FlutterGemma.installModel(
          modelType: model.modelType,
          fileType: model.fileType,
        )
        .fromNetwork(model.url, foreground: model.useForegroundDownload)
        .withProgress(onProgress)
        .install();
    _model = null;
    AppLogger.info('ai.install_model_network.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'hasActiveModel': hasActiveModel,
    });
  }

  Future<void> installEmbedderFromNetwork({
    required DownloadableEmbeddingModel model,
    required void Function(int modelProgress, int tokenizerProgress) onProgress,
  }) async {
    final stopwatch = Stopwatch()..start();
    AppLogger.info('ai.install_embedder_network.start', {
      'label': model.label,
      'size': model.size,
      'dimension': model.dimension,
      'maxSequenceLength': model.maxSequenceLength,
      'maxInputChars': model.maxInputChars,
    });
    await initialize();
    await _ensureEmbedderSupported();
    await _closeInferenceModel();
    var modelProgress = 0;
    var tokenizerProgress = 0;
    await FlutterGemma.installEmbedder()
        .modelFromNetwork(model.url)
        .tokenizerFromNetwork(
          model.tokenizerUrl,
          iosPath: model.iosTokenizerUrl,
        )
        .withModelProgress((progress) {
          modelProgress = progress;
          AppLogger.info('ai.install_embedder_network.model_progress', {
            'progress': progress,
            'tokenizerProgress': tokenizerProgress,
          });
          onProgress(modelProgress, tokenizerProgress);
        })
        .withTokenizerProgress((progress) {
          tokenizerProgress = progress;
          AppLogger.info('ai.install_embedder_network.tokenizer_progress', {
            'modelProgress': modelProgress,
            'progress': progress,
          });
          onProgress(modelProgress, tokenizerProgress);
        })
        .install();
    await _embedder?.close();
    _embedder = null;
    AppLogger.info('ai.install_embedder_network.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'hasActiveEmbedder': hasActiveEmbedder,
    });
  }

  Future<List<List<double>>> embedDocuments(
    List<String> texts, {
    int maxInputChars = 900,
  }) {
    return _withNativeAccess('embed_documents', () async {
      final stopwatch = Stopwatch()..start();
      AppLogger.info('ai.embed_documents.start', {
        'textCount': texts.length,
        'requestedMaxInputChars': maxInputChars,
        'safeMaxInputChars': _safeEmbeddingInputChars(maxInputChars),
        'textStats': AppLogger.textStats(texts),
      });
      await initialize();
      await _ensureEmbedderSupported();
      await _closeInferenceModel();
      final embedder = _embedder ??= await FlutterGemma.getActiveEmbedder(
        preferredBackend: PreferredBackend.cpu,
      );
      final embeddings = <List<double>>[];
      const batchSize = 8;

      for (var start = 0; start < texts.length; start += batchSize) {
        final end = start + batchSize > texts.length
            ? texts.length
            : start + batchSize;
        final batchTexts = texts.sublist(start, end);
        AppLogger.info('ai.embed_documents.batch.start', {
          'startIndex': start,
          'endIndexExclusive': end,
          'batchCount': batchTexts.length,
          'textStats': AppLogger.textStats(batchTexts),
          'sampleTexts': _textSamples(batchTexts),
        });
        final batchEmbeddings = await _generateDocumentEmbeddings(
          embedder,
          batchTexts,
          maxInputChars: maxInputChars,
        );
        AppLogger.info('ai.embed_documents.batch.done', {
          'startIndex': start,
          'batchCount': batchEmbeddings.length,
          'dimension': batchEmbeddings.isEmpty
              ? 0
              : batchEmbeddings.first.length,
        });
        embeddings.addAll(batchEmbeddings);
      }

      AppLogger.info('ai.embed_documents.done', {
        'embeddingCount': embeddings.length,
        'dimension': embeddings.isEmpty ? 0 : embeddings.first.length,
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      return embeddings;
    });
  }

  Future<List<double>> embedQuery(String text, {int maxInputChars = 450}) {
    return _withNativeAccess('embed_query', () async {
      final stopwatch = Stopwatch()..start();
      AppLogger.info('ai.embed_query.start', {
        'textChars': text.length,
        'requestedMaxInputChars': maxInputChars,
        'safeMaxInputChars': _safeEmbeddingInputChars(maxInputChars),
        'textPreview': AppLogger.preview(text),
      });
      await initialize();
      await _ensureEmbedderSupported();
      await _closeInferenceModel();
      final embedder = _embedder ??= await FlutterGemma.getActiveEmbedder(
        preferredBackend: PreferredBackend.cpu,
      );
      final embedding = await _generateQueryEmbedding(
        embedder,
        text,
        maxInputChars: maxInputChars,
      );
      AppLogger.info('ai.embed_query.done', {
        'dimension': embedding.length,
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      return embedding;
    });
  }

  @visibleForTesting
  String clipEmbeddingInputForTesting(String text, int maxChars) {
    return _clipEmbeddingInput(text, maxChars);
  }

  Future<String> answer({
    required String question,
    required List<SearchHit> hits,
    LocalInferenceConfig? config,
  }) {
    return _withNativeAccess('answer', () async {
      final stopwatch = Stopwatch()..start();
      AppLogger.info('ai.answer.start', {
        'questionChars': question.length,
        'questionPreview': AppLogger.preview(question),
        'hitCount': hits.length,
        'hits': _hitSummaries(hits),
      });
      await initialize();
      if (hits.isEmpty) {
        AppLogger.warn('ai.answer.no_hits', {
          'durationMs': stopwatch.elapsedMilliseconds,
        });
        return 'I could not find relevant local context for that question. Import more files or try a more specific query.';
      }

      if (!hasActiveModel) {
        final message =
            'The local answer engine is not installed yet. Install dependencies before asking questions.';
        AppLogger.warn('ai.answer.no_model', {
          'answerChars': message.length,
          'answerPreview': AppLogger.preview(message),
          'durationMs': stopwatch.elapsedMilliseconds,
        });
        return message;
      }

      final effectiveConfig = config ?? const LocalInferenceConfig();
      try {
        await _closeEmbedder();
        AppLogger.info('ai.answer.model.load.start', {
          'maxTokens': effectiveConfig.maxTokens,
          'preferredBackend': effectiveConfig.preferredBackend.name,
        });
        final model = _model ??= await FlutterGemma.getActiveModel(
          maxTokens: effectiveConfig.maxTokens,
          preferredBackend: effectiveConfig.preferredBackend,
        );
        AppLogger.info('ai.answer.model.load.done');
        final session = await model.createSession(
          temperature: effectiveConfig.temperature,
          topK: effectiveConfig.topK,
          topP: effectiveConfig.topP,
        );
        AppLogger.info('ai.answer.session.created', {
          'temperature': effectiveConfig.temperature,
          'topK': effectiveConfig.topK,
          'topP': effectiveConfig.topP,
        });
        try {
          final prompt = await _promptForSession(
            session: session,
            question: question,
            hits: hits,
            config: effectiveConfig,
          );
          AppLogger.info('ai.answer.prompt.ready', {
            'promptChars': prompt.length,
            'promptPreview': AppLogger.preview(prompt, 240),
          });
          await session.addQueryChunk(Message.text(text: prompt, isUser: true));
          AppLogger.info('ai.answer.generation.start');
          final response = await session.getResponse();
          final cleaned = _cleanGeneratedAnswer(response);
          if (cleaned.isEmpty) {
            const message =
                'The local model returned an empty answer. Try again with a more specific question.';
            AppLogger.warn('ai.answer.empty_response', {
              'answerChars': message.length,
              'durationMs': stopwatch.elapsedMilliseconds,
            });
            return message;
          }
          AppLogger.info('ai.answer.generation.done', {
            'rawResponseChars': response.length,
            'answerChars': cleaned.length,
            'answerPreview': AppLogger.preview(cleaned),
            'durationMs': stopwatch.elapsedMilliseconds,
          });
          return cleaned;
        } finally {
          await session.close();
          AppLogger.info('ai.answer.session.closed');
        }
      } on _PromptTooLongException catch (e) {
        AppLogger.warn('ai.answer.prompt_too_long', {'message': e.toString()});
        return e.toString();
      } catch (e, st) {
        AppLogger.error('ai.answer.model_error', e, st, {
          'durationMs': stopwatch.elapsedMilliseconds,
        });
        return 'The local model could not answer with the selected on-device model: $e';
      }
    });
  }

  Future<void> dispose() async {
    AppLogger.info('ai.dispose.start', {
      'hasCachedModel': _model != null,
      'hasCachedEmbedder': _embedder != null,
    });
    await _model?.close();
    await _embedder?.close();
    _model = null;
    _embedder = null;
    AppLogger.info('ai.dispose.done');
  }

  Future<T> _withNativeAccess<T>(
    String operation,
    Future<T> Function() action,
  ) async {
    final previous = _nativeOperation;
    final completer = Completer<void>();
    _nativeOperation = completer.future;
    final operationId = ++_nativeOperationSequence;
    final queueStopwatch = Stopwatch()..start();
    AppLogger.info('ai.native.queue', {
      'operationId': operationId,
      'operation': operation,
    });

    await previous;
    final operationStopwatch = Stopwatch()..start();
    AppLogger.info('ai.native.start', {
      'operationId': operationId,
      'operation': operation,
      'queuedMs': queueStopwatch.elapsedMilliseconds,
    });
    try {
      final result = await action();
      AppLogger.info('ai.native.done', {
        'operationId': operationId,
        'operation': operation,
        'durationMs': operationStopwatch.elapsedMilliseconds,
      });
      return result;
    } catch (e, st) {
      AppLogger.error('ai.native.error', e, st, {
        'operationId': operationId,
        'operation': operation,
        'durationMs': operationStopwatch.elapsedMilliseconds,
      });
      rethrow;
    } finally {
      completer.complete();
    }
  }

  Future<void> _closeInferenceModel() async {
    if (_model == null) return;
    AppLogger.info('ai.model.close.start');
    await _model?.close();
    _model = null;
    AppLogger.info('ai.model.close.done');
  }

  Future<void> _closeEmbedder() async {
    if (_embedder == null) return;
    AppLogger.info('ai.embedder.close.start');
    await _embedder?.close();
    _embedder = null;
    AppLogger.info('ai.embedder.close.done');
  }

  Future<void> _ensureEmbedderSupported() async {
    if (await supportsLocalEmbedder()) return;
    final abis = await supportedAbis();
    AppLogger.warn('ai.embedder.unsupported_abi', {'abis': abis});
    throw UnsupportedError(
      'The current Android device ABI is ${abis.join(', ')}. '
      'flutter_gemma embeddings require an ARM64 device because '
      'libgemma_embedding_model_jni.so is packaged for arm64-v8a.',
    );
  }

  @visibleForTesting
  String buildPromptForTesting({
    required String question,
    required List<SearchHit> hits,
    LocalInferenceConfig config = const LocalInferenceConfig(),
  }) {
    return _prompt(question, hits, _PromptBudget.fromConfig(config));
  }

  @visibleForTesting
  String? extractiveAnswerForTesting({
    required String question,
    required List<SearchHit> hits,
  }) {
    return _extractiveAnswer(question, hits);
  }

  Future<String> _promptForSession({
    required InferenceModelSession session,
    required String question,
    required List<SearchHit> hits,
    required LocalInferenceConfig config,
  }) async {
    final targetTokens = config.promptTokenTarget;
    final hardTokenLimit = config.hardPromptTokenLimit;
    String? smallestPrompt;
    int? smallestTokens;

    final budgets = _promptBudgets(config);
    for (var i = 0; i < budgets.length; i++) {
      final budget = budgets[i];
      final prompt = _prompt(question, hits, budget);
      final tokenCount = await session.sizeInTokens(prompt);
      smallestPrompt = prompt;
      smallestTokens = tokenCount;
      AppLogger.info('ai.prompt.budget_checked', {
        'attempt': i + 1,
        'promptChars': prompt.length,
        'tokenCount': tokenCount,
        'targetTokens': targetTokens,
        'hardTokenLimit': hardTokenLimit,
        'budget': budget.toLog(),
      });

      if (tokenCount <= targetTokens) {
        AppLogger.info('ai.prompt.budget_selected', {
          'attempt': i + 1,
          'tokenCount': tokenCount,
          'budget': budget.toLog(),
        });
        return prompt;
      }
    }

    if (smallestPrompt != null &&
        smallestTokens != null &&
        smallestTokens < hardTokenLimit) {
      AppLogger.warn('ai.prompt.using_smallest_under_hard_limit', {
        'tokenCount': smallestTokens,
        'hardTokenLimit': hardTokenLimit,
      });
      return smallestPrompt;
    }

    AppLogger.warn('ai.prompt.too_long', {
      'smallestTokens': smallestTokens,
      'hardTokenLimit': hardTokenLimit,
      'targetTokens': targetTokens,
    });
    throw _PromptTooLongException(
      'The selected local model only accepts about ${config.maxTokens} tokens. '
      'The question plus the smallest useful local context is still too long. '
      'Try a shorter question or select a larger model.',
    );
  }

  List<_PromptBudget> _promptBudgets(LocalInferenceConfig config) {
    return [
      _PromptBudget.fromConfig(config),
      _PromptBudget(
        maxHits: _minInt(config.maxPromptHits, 4),
        maxContextChars: _minInt(config.maxContextChars, 2200),
        maxExcerptChars: _minInt(config.maxExcerptChars, 560),
        maxQuestionChars: _minInt(config.maxQuestionChars, 900),
      ),
      _PromptBudget(
        maxHits: _minInt(config.maxPromptHits, 3),
        maxContextChars: _minInt(config.maxContextChars, 1400),
        maxExcerptChars: _minInt(config.maxExcerptChars, 420),
        maxQuestionChars: _minInt(config.maxQuestionChars, 700),
      ),
      _PromptBudget(
        maxHits: _minInt(config.maxPromptHits, 2),
        maxContextChars: _minInt(config.maxContextChars, 800),
        maxExcerptChars: _minInt(config.maxExcerptChars, 280),
        maxQuestionChars: _minInt(config.maxQuestionChars, 500),
      ),
      _PromptBudget(
        maxHits: 1,
        maxContextChars: _minInt(config.maxContextChars, 420),
        maxExcerptChars: _minInt(config.maxExcerptChars, 220),
        maxQuestionChars: _minInt(config.maxQuestionChars, 360),
      ),
      _PromptBudget(
        maxHits: 1,
        maxContextChars: _minInt(config.maxContextChars, 240),
        maxExcerptChars: _minInt(config.maxExcerptChars, 160),
        maxQuestionChars: _minInt(config.maxQuestionChars, 260),
      ),
    ];
  }

  String _prompt(String question, List<SearchHit> hits, _PromptBudget budget) {
    final contextHits = _contextHits(hits, maxHits: budget.maxHits);
    AppLogger.info('ai.prompt.context_hits.selected', {
      'requestedMaxHits': budget.maxHits,
      'selectedCount': contextHits.length,
      'hits': _hitSummaries(contextHits),
    });
    final buffer = StringBuffer()
      ..writeln('Answer using only these local excerpts.')
      ..writeln(
        'Copy exact names, places, dates, and amounts from the matching excerpt.',
      )
      ..writeln(
        'If the exact answer is missing, say: Not found in the local excerpts.',
      )
      ..writeln('Return one or two concise sentences.')
      ..writeln('End with the source file name in brackets, like [notes.md].')
      ..writeln(
        'Do not cite excerpt numbers. Do not output only a citation or a raw excerpt.',
      )
      ..writeln()
      ..writeln('Excerpts:');

    var used = 0;
    for (var i = 0; i < contextHits.length; i++) {
      final hit = contextHits[i];
      final remaining = budget.maxContextChars - used;
      if (remaining <= 0) break;
      final excerpt = _clipText(
        _focusedContext(question, hit),
        _minInt(remaining, budget.maxExcerptChars),
      );
      if (excerpt.isEmpty) continue;
      used += excerpt.length;
      final source = _clipText(hit.document.name, 80);
      buffer
        ..writeln()
        ..writeln('Excerpt ${i + 1}')
        ..writeln('Source file: $source')
        ..writeln('Chunk: ${hit.chunk.index + 1}')
        ..writeln(excerpt);
    }

    buffer
      ..writeln()
      ..writeln('Question: ${_clipText(question, budget.maxQuestionChars)}')
      ..writeln('Answer:');
    return buffer.toString();
  }

  String? _extractiveAnswer(String question, List<SearchHit> hits) {
    final identifiers = _identifiers(question);
    final fields = _requestedFields(question);
    if (fields.isEmpty) return null;

    AppLogger.info('ai.extractive.start', {
      'identifiers': identifiers,
      'fields': fields.map((field) => field.key).toList(growable: false),
      'hitCount': hits.length,
    });

    if (identifiers.isEmpty) {
      final answer = _fieldOnlyAnswer(question, hits, fields);
      if (answer != null) return answer;
      AppLogger.info('ai.extractive.no_identifier_field_match', {
        'fields': fields.map((field) => field.key).toList(growable: false),
      });
      return null;
    }

    var sawIdentifier = false;
    for (final identifier in identifiers) {
      for (final section in _identifierSections(identifier, hits)) {
        sawIdentifier = true;
        final values = <String>[];
        var hasAllFields = true;
        for (final field in fields) {
          final value = _fieldValue(section.text, field.labels);
          if (value == null || value.isEmpty) {
            hasAllFields = false;
            break;
          }
          values.add('${field.outputLabel}: $value');
        }
        if (hasAllFields) {
          final answer = '${values.join('. ')}. [${section.hit.document.name}]';
          AppLogger.info('ai.extractive.match', {
            'identifier': identifier,
            'document': section.hit.document.name,
            'chunkIndex': section.hit.chunk.index,
            'answerPreview': AppLogger.preview(answer),
          });
          return answer;
        }
      }
    }

    if (!sawIdentifier) {
      final answer =
          '${identifiers.first} is missing from the local context. '
          '[${hits.first.document.name}]';
      AppLogger.warn('ai.extractive.identifier_missing', {
        'identifier': identifiers.first,
        'answerPreview': AppLogger.preview(answer),
      });
      return answer;
    }

    AppLogger.info('ai.extractive.no_complete_field_match', {
      'identifiers': identifiers,
      'fields': fields.map((field) => field.key).toList(growable: false),
    });
    return null;
  }

  String? _fieldOnlyAnswer(
    String question,
    List<SearchHit> hits,
    List<_RequestedField> fields,
  ) {
    if (fields.length > 2) return null;
    for (final hit in hits.take(4)) {
      final section = _focusedContext(question, hit);
      final values = <String>[];
      for (final field in fields) {
        final value =
            _fieldValue(section, field.labels) ??
            _specialFieldValue(question, section, field, hit);
        if (value != null && value.isNotEmpty) {
          values.add('${field.outputLabel}: $value');
        }
      }
      if (values.length == fields.length) {
        final answer = '${values.join('. ')}. [${hit.document.name}]';
        AppLogger.info('ai.extractive.field_only_match', {
          'document': hit.document.name,
          'chunkIndex': hit.chunk.index,
          'fields': fields.map((field) => field.key).toList(growable: false),
          'answerPreview': AppLogger.preview(answer),
        });
        return answer;
      }
    }
    return null;
  }

  String? _specialFieldValue(
    String question,
    String text,
    _RequestedField field,
    SearchHit hit,
  ) {
    if (field.key == 'company' &&
        _looksLikeOfferLetterQuestion(question, hit)) {
      return _companyFromOfferLetter(text);
    }
    return null;
  }

  Iterable<_IdentifierSection> _identifierSections(
    String identifier,
    List<SearchHit> hits,
  ) sync* {
    final seen = <String>{};
    for (final hit in hits) {
      final section = _sectionAroundIdentifier(hit.contextText, identifier);
      if (section != null && seen.add(section)) {
        yield _IdentifierSection(hit: hit, text: section);
      }
    }
    for (final hit in hits) {
      final section = _sectionAroundIdentifier(hit.chunk.content, identifier);
      if (section != null && seen.add(section)) {
        yield _IdentifierSection(hit: hit, text: section);
      }
    }
  }

  String _focusedContext(String question, SearchHit hit) {
    final identifiers = _identifiers(question);
    for (final identifier in identifiers) {
      final section = _sectionAroundIdentifier(hit.contextText, identifier);
      if (section != null) return section;
      final chunkSection = _sectionAroundIdentifier(
        hit.chunk.content,
        identifier,
      );
      if (chunkSection != null) return chunkSection;
    }

    return hit.contextText;
  }

  List<String> _identifiers(String question) {
    final identifiers = <String>{};
    final matches = RegExp(
      r'\b[A-Z][A-Z0-9]{1,12}[-_][A-Z0-9][A-Z0-9-_]{1,24}\b',
      caseSensitive: false,
    ).allMatches(question);
    for (final match in matches) {
      identifiers.add(match.group(0)!.toUpperCase());
    }
    identifiers.addAll(_naturalLanguageIdentifiers(question));
    return identifiers.toList(growable: false);
  }

  List<String> _naturalLanguageIdentifiers(String question) {
    final lower = question.toLowerCase();
    final identifiers = <String>{};
    void addCandidate(String value) {
      final candidate = value.replaceAll(RegExp(r"^[.'-]+|[.'-]+$"), '');
      if (candidate.length >= 3 && !_queryStopWords.contains(candidate)) {
        identifiers.add(candidate);
      }
    }

    for (final match in RegExp(
      r"\b(?:for|of|about|regarding)\s+([a-z][a-z.'-]{2,})\b",
    ).allMatches(lower)) {
      addCandidate(match.group(1)!.trim());
    }

    for (final match in RegExp(
      r"\b([a-z][a-z.'-]{2,})(?:'s)?\s+(?:job\s+location|location|salary|ctc|role|position|joining\s+date)\b",
    ).allMatches(lower)) {
      addCandidate(match.group(1)!.trim());
    }

    return identifiers.toList(growable: false);
  }

  List<_RequestedField> _requestedFields(String question) {
    final lower = question.toLowerCase();
    final fields = <_RequestedField>[];

    void addIf(bool condition, _RequestedField field) {
      if (condition && !fields.any((existing) => existing.key == field.key)) {
        fields.add(field);
      }
    }

    addIf(
      lower.contains('owner') || lower.contains('team'),
      _RequestedField(
        key: 'owner',
        outputLabel: 'Owner team',
        labels: const ['Owner team', 'Owner'],
      ),
    );
    addIf(
      lower.contains('approval'),
      _RequestedField(
        key: 'approval',
        outputLabel: 'Approval path',
        labels: const ['Approval path', 'Approval'],
      ),
    );
    addIf(
      lower.contains('region'),
      _RequestedField(
        key: 'region',
        outputLabel: 'Primary region',
        labels: const ['Primary region', 'Region'],
      ),
    );
    addIf(
      lower.contains('location') ||
          lower.contains('workplace') ||
          lower.contains('posting') ||
          lower.contains('place of work') ||
          lower.contains('city'),
      _RequestedField(
        key: 'location',
        outputLabel: 'Location',
        labels: const [
          'Job location',
          'Work location',
          'Office location',
          'Place of posting',
          'Place of work',
          'Location',
          'City',
        ],
      ),
    );
    addIf(
      lower.contains('company') ||
          lower.contains('employer') ||
          lower.contains('organization') ||
          lower.contains('organisation') ||
          lower.contains('issued') ||
          lower.contains('issuer'),
      _RequestedField(
        key: 'company',
        outputLabel: 'Company',
        labels: const [
          'Company',
          'Employer',
          'Organization',
          'Organisation',
          'Issued by',
          'Issuer',
          'From',
        ],
      ),
    );
    addIf(
      lower.contains('role') ||
          lower.contains('position') ||
          lower.contains('designation') ||
          lower.contains('job title') ||
          lower.contains('title'),
      _RequestedField(
        key: 'role',
        outputLabel: 'Role',
        labels: const ['Role', 'Position', 'Designation', 'Job title', 'Title'],
      ),
    );
    addIf(
      lower.contains('salary') ||
          lower.contains('ctc') ||
          lower.contains('compensation') ||
          lower.contains('package') ||
          lower.contains('pay'),
      _RequestedField(
        key: 'salary',
        outputLabel: 'Compensation',
        labels: const ['Salary', 'CTC', 'Compensation', 'Package', 'Pay'],
      ),
    );
    addIf(
      lower.contains('joining') ||
          lower.contains('start date') ||
          lower.contains('date of joining'),
      _RequestedField(
        key: 'joining_date',
        outputLabel: 'Joining date',
        labels: const ['Joining date', 'Date of joining', 'Start date'],
      ),
    );
    addIf(
      lower.contains('deadline') ||
          lower.contains('due date') ||
          lower.contains(' date'),
      _RequestedField(
        key: 'deadline',
        outputLabel: 'Deadline',
        labels: const ['Deadline', 'Due date'],
      ),
    );
    addIf(
      lower.contains('budget') ||
          lower.contains('amount') ||
          lower.contains('cost') ||
          lower.contains(' cap'),
      _RequestedField(
        key: 'budget',
        outputLabel: 'Budget cap',
        labels: const ['Budget cap', 'Budget', 'Amount'],
      ),
    );
    addIf(
      lower.contains('retention') || lower.contains('window'),
      _RequestedField(
        key: 'retention',
        outputLabel: 'Retention window',
        labels: const ['Retention window', 'Retention'],
      ),
    );
    addIf(
      lower.contains('severity'),
      _RequestedField(
        key: 'severity',
        outputLabel: 'Severity class',
        labels: const ['Severity class', 'Severity'],
      ),
    );
    addIf(
      lower.contains('codename') || lower.contains('project name'),
      _RequestedField(
        key: 'codename',
        outputLabel: 'Project codename',
        labels: const ['Project codename', 'Codename'],
      ),
    );

    return fields;
  }

  String? _fieldValue(String section, List<String> labels) {
    for (final label in labels) {
      final match = RegExp(
        '${RegExp.escape(label)}\\s*[:=-]\\s*([^\\n;]+)',
        caseSensitive: false,
      ).firstMatch(section);
      final value = match?.group(1)?.trim();
      if (value != null && value.isNotEmpty) {
        return _cleanFieldValue(value);
      }
    }
    return null;
  }

  String _cleanFieldValue(String value) {
    return value
        .replaceAll(RegExp(r'\s+'), ' ')
        .replaceAll(RegExp(r'\s*[.,:;]+$'), '')
        .trim();
  }

  bool _looksLikeOfferLetterQuestion(String question, SearchHit hit) {
    final lowerQuestion = question.toLowerCase();
    final lowerName = hit.document.name.toLowerCase();
    return lowerQuestion.contains('offer') ||
        lowerQuestion.contains('issued') ||
        lowerName.contains('offer');
  }

  String? _companyFromOfferLetter(String text) {
    for (final pattern in [
      r'\bissued\s+by\s*[:=-]?\s*([^\n.;]{2,100})',
      r'\bemployer\s*[:=-]\s*([^\n.;]{2,100})',
      r'\bcompany\s*[:=-]\s*([^\n.;]{2,100})',
      r'\bfrom\s*[:=-]\s*([^\n.;]{2,100})',
    ]) {
      final match = RegExp(pattern, caseSensitive: false).firstMatch(text);
      final value = match?.group(1);
      if (value != null) {
        final cleaned = _cleanCompanyName(value);
        if (_looksLikeCompanyName(cleaned)) return cleaned;
      }
    }

    final lines = text
        .split(RegExp(r'\r?\n'))
        .map(_cleanCompanyName)
        .where((line) => line.isNotEmpty)
        .take(16);
    for (final line in lines) {
      if (_looksLikeCompanyName(line)) return line;
    }
    return null;
  }

  String _cleanCompanyName(String value) {
    return value
        .replaceAll(RegExp(r'^[^\w]+'), '')
        .replaceAll(RegExp(r'\s+'), ' ')
        .replaceAll(RegExp(r'\s*[.,:;]+$'), '')
        .trim();
  }

  bool _looksLikeCompanyName(String value) {
    if (value.length < 3 || value.length > 90) return false;
    final lower = value.toLowerCase();
    if (RegExp(
      r'\b(offer|letter|date|dear|subject|sub|ref|page|phone|email|candidate|employee|human resources|department)\b',
    ).hasMatch(lower)) {
      return false;
    }
    if (RegExp(
      r'\b(pvt|private|ltd|limited|llp|inc|corp|corporation|technologies|technology|solutions|systems|services|consulting|bank|labs|software|industries)\b',
    ).hasMatch(lower)) {
      return true;
    }
    final words = value.split(RegExp(r'\s+')).where((word) => word.isNotEmpty);
    if (words.length < 2 || words.length > 8) return false;
    return RegExp(r'^[A-Z][A-Za-z0-9&().,\- ]+$').hasMatch(value);
  }

  String? _sectionAroundIdentifier(String text, String identifier) {
    final escaped = RegExp.escape(identifier);
    final heading = RegExp(
      '^#{1,6}\\s+.*\\b$escaped\\b.*\$',
      caseSensitive: false,
      multiLine: true,
    ).firstMatch(text);
    if (heading != null) {
      final nextHeading = RegExp(
        '^#{1,6}\\s+',
        multiLine: true,
      ).firstMatch(text.substring(heading.end));
      final end = nextHeading == null
          ? text.length
          : heading.end + nextHeading.start;
      return text.substring(heading.start, end).trim();
    }

    final match = RegExp(
      r'\b' + escaped + r'\b',
      caseSensitive: false,
    ).firstMatch(text);
    if (match == null) return null;

    final start = _nearestBoundaryBefore(text, match.start);
    final end = _nearestBoundaryAfter(text, match.end);
    return text.substring(start, end).trim();
  }

  int _nearestBoundaryBefore(String text, int index) {
    final heading = text.lastIndexOf(RegExp(r'\n#{1,6}\s+'), index);
    if (heading != -1) return heading + 1;
    final paragraph = text.lastIndexOf('\n\n', index);
    if (paragraph != -1) return paragraph + 2;
    final sentence = text.lastIndexOf(RegExp(r'[.!?]\s+'), index);
    if (sentence != -1) return sentence + 2;
    return 0;
  }

  int _nearestBoundaryAfter(String text, int index) {
    final rest = text.substring(index);
    final heading = RegExp(r'\n#{1,6}\s+').firstMatch(rest);
    if (heading != null) return index + heading.start;
    final paragraph = rest.indexOf('\n\n');
    if (paragraph != -1) return index + paragraph;
    final sentence = RegExp(r'[.!?]\s+').firstMatch(rest);
    if (sentence != null) return index + sentence.end;
    return text.length;
  }

  String _cleanGeneratedAnswer(String answer) {
    return answer
        .replaceFirst(
          RegExp(r'^\s*(?:answer|assistant)\s*:\s*', caseSensitive: false),
          '',
        )
        .trim();
  }

  List<SearchHit> _contextHits(List<SearchHit> hits, {required int maxHits}) {
    final selected = <SearchHit>[];
    final seenChunks = <String>{};
    final perDocument = <String, int>{};
    final hitLimit = maxHits < 1 ? 1 : maxHits;

    for (final cap in [1, 2, 4]) {
      for (final hit in hits) {
        if (selected.length >= hitLimit) return selected;
        if (seenChunks.contains(hit.chunk.id)) continue;

        final count = perDocument[hit.document.id] ?? 0;
        if (count >= cap) continue;

        selected.add(hit);
        seenChunks.add(hit.chunk.id);
        perDocument[hit.document.id] = count + 1;
      }
    }

    return selected;
  }

  String _clipText(String text, int maxChars) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.length <= maxChars) return normalized;
    if (maxChars <= 3) return normalized.substring(0, maxChars);
    return '${normalized.substring(0, maxChars - 3).trimRight()}...';
  }

  String _clipEmbeddingInput(String text, int maxChars) {
    final limit = _safeEmbeddingInputChars(maxChars);
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.length <= limit) return normalized;
    return normalized.substring(0, limit).trimRight();
  }

  Future<List<List<double>>> _generateDocumentEmbeddings(
    EmbeddingModel embedder,
    List<String> texts, {
    required int maxInputChars,
  }) async {
    var limit = _safeEmbeddingInputChars(maxInputChars);
    while (true) {
      try {
        final clippedTexts = texts
            .map((text) => _clipEmbeddingInput(text, limit))
            .toList(growable: false);
        AppLogger.info('ai.embed_documents.native_attempt', {
          'batchCount': texts.length,
          'limit': limit,
          'clippedStats': AppLogger.textStats(clippedTexts),
        });
        final embeddings = await embedder.generateEmbeddings(
          clippedTexts,
          taskType: TaskType.retrievalDocument,
        );
        AppLogger.info('ai.embed_documents.native_success', {
          'batchCount': embeddings.length,
          'dimension': embeddings.isEmpty ? 0 : embeddings.first.length,
          'limit': limit,
        });
        return embeddings;
      } on PlatformException catch (e) {
        final nextLimit = _nextEmbeddingRetryLimit(limit, e);
        if (nextLimit == null) rethrow;
        AppLogger.warn('ai.embed_documents.retry_with_shorter_input', {
          'previousLimit': limit,
          'nextLimit': nextLimit,
          'error': e.message ?? e.code,
        });
        limit = nextLimit;
      }
    }
  }

  Future<List<double>> _generateQueryEmbedding(
    EmbeddingModel embedder,
    String text, {
    required int maxInputChars,
  }) async {
    var limit = _safeEmbeddingInputChars(maxInputChars);
    while (true) {
      try {
        final clippedText = _clipEmbeddingInput(text, limit);
        AppLogger.info('ai.embed_query.native_attempt', {
          'limit': limit,
          'clippedChars': clippedText.length,
          'clippedPreview': AppLogger.preview(clippedText),
        });
        final embedding = await embedder.generateEmbedding(
          clippedText,
          taskType: TaskType.retrievalQuery,
        );
        AppLogger.info('ai.embed_query.native_success', {
          'dimension': embedding.length,
          'limit': limit,
        });
        return embedding;
      } on PlatformException catch (e) {
        final nextLimit = _nextEmbeddingRetryLimit(limit, e);
        if (nextLimit == null) rethrow;
        AppLogger.warn('ai.embed_query.retry_with_shorter_input', {
          'previousLimit': limit,
          'nextLimit': nextLimit,
          'error': e.message ?? e.code,
        });
        limit = nextLimit;
      }
    }
  }

  int _safeEmbeddingInputChars(int requestedMaxChars) {
    if (requestedMaxChars < 1) return 1;
    return requestedMaxChars < _maxSafeEmbeddingInputChars
        ? requestedMaxChars
        : _maxSafeEmbeddingInputChars;
  }

  int? _nextEmbeddingRetryLimit(int currentLimit, PlatformException error) {
    if (!_isEmbeddingLengthFailure(error)) return null;
    if (currentLimit <= _minEmbeddingRetryInputChars) return null;

    final reduced = (currentLimit * 3 / 4).floor();
    if (reduced < _minEmbeddingRetryInputChars) {
      return _minEmbeddingRetryInputChars;
    }
    return reduced < currentLimit ? reduced : null;
  }

  bool _isEmbeddingLengthFailure(PlatformException error) {
    final message = '${error.code} ${error.message} ${error.details}'
        .toLowerCase();
    return message.contains('max_input_size') ||
        message.contains('tokens.size') ||
        message.contains('max input');
  }

  int _minInt(int a, int b) => a < b ? a : b;

  List<Map<String, Object?>> _textSamples(List<String> texts) {
    return [
      for (var i = 0; i < texts.length && i < 3; i++)
        {
          'index': i,
          'chars': texts[i].length,
          'preview': AppLogger.preview(texts[i], 140),
        },
    ];
  }

  List<Map<String, Object?>> _hitSummaries(List<SearchHit> hits) {
    return [
      for (var i = 0; i < hits.length && i < 8; i++)
        {
          'rank': i + 1,
          'document': hits[i].document.name,
          'chunkIndex': hits[i].chunk.index,
          'score': AppLogger.score(hits[i].score),
          'keywordScore': AppLogger.score(hits[i].keywordScore),
          'vectorScore': AppLogger.score(hits[i].vectorScore),
          'chunkChars': hits[i].chunk.content.length,
          'contextChars': hits[i].contextText.length,
          'preview': AppLogger.preview(hits[i].contextText, 140),
        },
    ];
  }
}

class _PromptBudget {
  const _PromptBudget({
    required this.maxHits,
    required this.maxContextChars,
    required this.maxExcerptChars,
    required this.maxQuestionChars,
  });

  factory _PromptBudget.fromConfig(LocalInferenceConfig config) {
    return _PromptBudget(
      maxHits: config.maxPromptHits,
      maxContextChars: config.maxContextChars,
      maxExcerptChars: config.maxExcerptChars,
      maxQuestionChars: config.maxQuestionChars,
    );
  }

  final int maxHits;
  final int maxContextChars;
  final int maxExcerptChars;
  final int maxQuestionChars;

  Map<String, Object?> toLog() {
    return {
      'maxHits': maxHits,
      'maxContextChars': maxContextChars,
      'maxExcerptChars': maxExcerptChars,
      'maxQuestionChars': maxQuestionChars,
    };
  }
}

class _PromptTooLongException implements Exception {
  const _PromptTooLongException(this.message);

  final String message;

  @override
  String toString() => message;
}

class _IdentifierSection {
  const _IdentifierSection({required this.hit, required this.text});

  final SearchHit hit;
  final String text;
}

class _RequestedField {
  const _RequestedField({
    required this.key,
    required this.outputLabel,
    required this.labels,
  });

  final String key;
  final String outputLabel;
  final List<String> labels;
}
