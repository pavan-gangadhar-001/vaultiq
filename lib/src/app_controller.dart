import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

import 'model_catalog.dart';
import 'models.dart';
import 'services/app_logger.dart';
import 'services/document_store.dart';
import 'services/import_service.dart';
import 'services/local_ai_service.dart';
import 'services/text_extractor.dart';

class AppController extends ChangeNotifier {
  AppController({
    required this.store,
    required this.importService,
    required this.extractor,
    required this.aiService,
  });

  final DocumentStore store;
  final ImportService importService;
  final TextExtractor extractor;
  final LocalAiService aiService;

  final List<ChatTurn> messages = [
    ChatTurn.assistant(
      'Welcome to VaultIQ. Import local files or a folder, then ask questions. Everything is indexed on this device.',
    ),
  ];

  List<IndexedDocument> documents = [];
  IndexStats stats = const IndexStats.empty();
  bool isInitializing = true;
  bool isIndexing = false;
  bool isAnswering = false;
  bool isModelBusy = false;
  bool isEmbeddingBusy = false;
  bool hasLocalModel = false;
  bool hasEmbeddingModel = false;
  bool supportsEmbeddingModel = true;
  int? modelDownloadProgress;
  int? embeddingModelDownloadProgress;
  int? embeddingTokenizerDownloadProgress;
  List<String> supportedAbis = const [];
  String status = 'Opening local index';
  String? error;
  DownloadableModel selectedModel = DownloadableModel.qwen25OnePointFiveB;
  DownloadableEmbeddingModel selectedEmbeddingModel =
      DownloadableEmbeddingModel.gecko256;
  int _askSequence = 0;

  bool get isInstallingDependencies => isModelBusy || isEmbeddingBusy;

  bool get dependenciesReady {
    return hasLocalModel && (!supportsEmbeddingModel || hasEmbeddingModel);
  }

  bool get needsSetup => !isInitializing && !dependenciesReady;

  Future<void> initialize() async {
    final stopwatch = Stopwatch()..start();
    AppLogger.info('app.initialize.start');
    try {
      await store.open();
      await aiService.initialize();
      hasLocalModel = await aiService.activateInstalledModel(selectedModel);
      supportedAbis = await aiService.supportedAbis();
      selectedEmbeddingModel = aiService.embeddingModelForAbis(supportedAbis);
      supportsEmbeddingModel = await aiService.supportsEmbeddingModel(
        selectedEmbeddingModel,
      );
      hasEmbeddingModel =
          supportsEmbeddingModel &&
          await aiService.activateInstalledEmbedder(selectedEmbeddingModel);
      await _refreshIndexState();
      status = dependenciesReady ? 'Ready' : 'Install local AI to begin';
      AppLogger.info('app.initialize.done', {
        'durationMs': stopwatch.elapsedMilliseconds,
        'hasAnswerEngine': hasLocalModel,
        'hasSemanticSearch': hasEmbeddingModel,
        'semanticSearchModel': selectedEmbeddingModel.id,
        'semanticSearchRuntime': selectedEmbeddingModel.runtime.name,
        'supportsSemanticSearch': supportsEmbeddingModel,
        'supportedAbis': supportedAbis,
        'dependenciesReady': dependenciesReady,
        'stats': _statsLog(),
      });
    } catch (e, st) {
      AppLogger.error('app.initialize.error', e, st, {
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      error = 'Startup failed: $e';
      status = 'Needs attention';
    } finally {
      isInitializing = false;
      notifyListeners();
    }
  }

  Future<void> importFiles() async {
    if (isIndexing) return;
    AppLogger.info('import.files.pick.start');
    final picked = await importService.pickFiles();
    AppLogger.info('import.files.pick.done', {
      'count': picked.length,
      'items': _candidateSummaries(picked),
    });
    await _indexCandidates(picked, emptyMessage: 'No files selected');
  }

  Future<void> importFolder() async {
    if (isIndexing) return;
    AppLogger.info('import.folder.pick.start');
    final picked = await importService.pickFolder();
    AppLogger.info('import.folder.pick.done', {
      'count': picked.length,
      'items': _candidateSummaries(picked),
    });
    await _indexCandidates(picked, emptyMessage: 'No folder selected');
  }

  Future<void> _indexCandidates(
    List<ImportCandidate> candidates, {
    required String emptyMessage,
  }) async {
    if (candidates.isEmpty) {
      AppLogger.warn('index.skip.empty_selection', {'message': emptyMessage});
      status = emptyMessage;
      notifyListeners();
      return;
    }

    final stopwatch = Stopwatch()..start();
    AppLogger.info('index.batch.start', {
      'candidateCount': candidates.length,
      'hasSemanticSearch': hasEmbeddingModel,
      'candidates': _candidateSummaries(candidates),
    });
    isIndexing = true;
    error = null;
    var indexed = 0;
    var skipped = 0;
    status =
        'Preparing ${candidates.length} item${candidates.length == 1 ? '' : 's'}';
    notifyListeners();

    for (final candidate in candidates) {
      final candidateStopwatch = Stopwatch()..start();
      AppLogger.info('index.candidate.start', _candidateSummary(candidate));
      status = 'Indexing ${candidate.name}';
      notifyListeners();

      try {
        if (!extractor.isSupported(candidate.name)) {
          skipped++;
          AppLogger.warn('index.candidate.skip_unsupported', {
            ..._candidateSummary(candidate),
            'durationMs': candidateStopwatch.elapsedMilliseconds,
          });
          continue;
        }
        final extracted = await extractor.extract(candidate);
        AppLogger.info('index.candidate.extracted', {
          ..._candidateSummary(candidate),
          'bytes': extracted.bytes,
          'textChars': extracted.text.length,
          'textPreview': AppLogger.preview(extracted.text),
          'durationMs': candidateStopwatch.elapsedMilliseconds,
        });
        if (extracted.text.trim().isEmpty) {
          skipped++;
          AppLogger.warn('index.candidate.skip_empty_text', {
            ..._candidateSummary(candidate),
            'durationMs': candidateStopwatch.elapsedMilliseconds,
          });
          continue;
        }
        await store.upsertDocument(
          extracted,
          embedChunks: hasEmbeddingModel
              ? (chunks) => _embedChunksForIndexing(chunks, candidate.name)
              : null,
          embeddingModel: hasEmbeddingModel ? selectedEmbeddingModel.id : null,
        );
        indexed++;
        AppLogger.info('index.candidate.done', {
          ..._candidateSummary(candidate),
          'indexed': true,
          'durationMs': candidateStopwatch.elapsedMilliseconds,
        });
      } catch (e, st) {
        skipped++;
        AppLogger.error('index.candidate.error', e, st, {
          ..._candidateSummary(candidate),
          'durationMs': candidateStopwatch.elapsedMilliseconds,
        });
        error = 'Skipped ${candidate.name}: $e';
      }
    }

    await _refreshIndexState();
    isIndexing = false;
    status =
        'Indexed $indexed item${indexed == 1 ? '' : 's'}'
        '${skipped == 0 ? '' : ', skipped $skipped'}';
    AppLogger.info('index.batch.done', {
      'indexed': indexed,
      'skipped': skipped,
      'durationMs': stopwatch.elapsedMilliseconds,
      'stats': _statsLog(),
    });
    notifyListeners();
  }

  Future<void> ask(String question) async {
    final cleaned = question.trim();
    final askId = ++_askSequence;
    final stopwatch = Stopwatch()..start();
    AppLogger.info('ask.start', {
      'askId': askId,
      'questionChars': cleaned.length,
      'questionPreview': AppLogger.preview(cleaned),
      'isAnswering': isAnswering,
      'isIndexing': isIndexing,
      'isInstallingDependencies': isInstallingDependencies,
      'dependenciesReady': dependenciesReady,
      'stats': _statsLog(),
    });
    if (cleaned.isEmpty || isAnswering) {
      AppLogger.warn('ask.skip_unavailable', {
        'askId': askId,
        'reason': cleaned.isEmpty ? 'empty_question' : 'answer_in_progress',
      });
      return;
    }
    if (isInstallingDependencies || isIndexing) {
      status = isIndexing
          ? 'Finish indexing before asking'
          : 'Finishing local AI setup';
      AppLogger.warn('ask.skip_busy', {
        'askId': askId,
        'isIndexing': isIndexing,
        'isInstallingDependencies': isInstallingDependencies,
      });
      notifyListeners();
      return;
    }
    if (!dependenciesReady) {
      status = 'Install local AI to begin';
      AppLogger.warn('ask.skip_missing_dependencies', {
        'askId': askId,
        'hasAnswerEngine': hasLocalModel,
        'hasSemanticSearch': hasEmbeddingModel,
        'supportsSemanticSearch': supportsEmbeddingModel,
      });
      notifyListeners();
      return;
    }

    messages.add(ChatTurn.user(cleaned));
    final pending = ChatTurn.assistant('', pending: true);
    messages.add(pending);
    isAnswering = true;
    status = 'Searching local index';
    notifyListeners();

    try {
      status = 'Searching keyword index';
      notifyListeners();
      var hits = await store.search(cleaned, topK: 8);
      AppLogger.info('ask.keyword_search.done', {
        'askId': askId,
        'hitCount': hits.length,
        'hits': _hitSummaries(hits),
      });

      List<double>? queryEmbedding;
      final shouldUseSemanticSearch =
          supportsEmbeddingModel &&
          hasEmbeddingModel &&
          stats.embeddedChunkCount > 0 &&
          _shouldUseSemanticSearch(hits);
      AppLogger.info('ask.semantic_decision', {
        'askId': askId,
        'shouldUseSemanticSearch': shouldUseSemanticSearch,
        'supportsSemanticSearch': supportsEmbeddingModel,
        'hasSemanticSearch': hasEmbeddingModel,
        'embeddedChunkCount': stats.embeddedChunkCount,
        'keywordHitCount': hits.length,
        'topKeywordScore': hits.isEmpty
            ? 0
            : AppLogger.score(hits.first.keywordScore),
      });
      if (shouldUseSemanticSearch) {
        status = 'Embedding question';
        notifyListeners();
        try {
          queryEmbedding = await aiService.embedQuery(
            cleaned,
            maxInputChars: selectedEmbeddingModel.maxInputChars,
          );
          AppLogger.info('ask.query_embedding.done', {
            'askId': askId,
            'dimension': queryEmbedding.length,
          });
        } catch (e, st) {
          AppLogger.error('ask.query_embedding.error', e, st, {'askId': askId});
          error = 'Semantic search skipped: ${_describeFailure(e)}';
        }
      }

      if (queryEmbedding != null) {
        status = 'Searching hybrid index';
        notifyListeners();
        hits = await store.search(
          cleaned,
          topK: 8,
          queryEmbedding: queryEmbedding,
        );
        AppLogger.info('ask.hybrid_search.done', {
          'askId': askId,
          'hitCount': hits.length,
          'hits': _hitSummaries(hits),
        });
      }
      status = hasLocalModel ? 'Running local model' : 'Composing local answer';
      notifyListeners();

      AppLogger.info('ask.answer.start', {
        'askId': askId,
        'hitCount': hits.length,
        'hits': _hitSummaries(hits),
      });
      final answer = await aiService.answer(
        question: cleaned,
        hits: hits,
        config: selectedModel.inferenceConfig,
      );
      final index = messages.indexWhere((turn) => turn.id == pending.id);
      if (index != -1) {
        messages[index] = ChatTurn.assistant(answer, sources: hits);
      }
      status = 'Ready';
      AppLogger.info('ask.done', {
        'askId': askId,
        'answerChars': answer.length,
        'answerPreview': AppLogger.preview(answer),
        'sourceCount': hits.length,
        'durationMs': stopwatch.elapsedMilliseconds,
      });
    } catch (e, st) {
      AppLogger.error('ask.error', e, st, {
        'askId': askId,
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      final index = messages.indexWhere((turn) => turn.id == pending.id);
      if (index != -1) {
        messages[index] = ChatTurn.assistant('I could not answer that: $e');
      }
      error = 'Answer failed: $e';
      status = 'Needs attention';
    } finally {
      isAnswering = false;
      notifyListeners();
    }
  }

  Future<void> installDependencies() async {
    if (isInstallingDependencies) return;
    final stopwatch = Stopwatch()..start();
    AppLogger.info('setup.install_dependencies.start', {
      'hasAnswerEngine': hasLocalModel,
      'hasSemanticSearch': hasEmbeddingModel,
      'supportsSemanticSearch': supportsEmbeddingModel,
    });
    error = null;

    if (!hasLocalModel) {
      await installSelectedModel();
      if (!hasLocalModel) return;
    }

    final missingEmbeddings = supportsEmbeddingModel
        ? await store.countChunksMissingEmbeddings()
        : 0;
    if (supportsEmbeddingModel &&
        (!hasEmbeddingModel || missingEmbeddings > 0)) {
      await installEmbeddingModel();
      if (!hasEmbeddingModel) return;
    }

    status = 'Local AI ready';
    AppLogger.info('setup.install_dependencies.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'hasAnswerEngine': hasLocalModel,
      'hasSemanticSearch': hasEmbeddingModel,
      'dependenciesReady': dependenciesReady,
    });
    notifyListeners();
  }

  bool _shouldUseSemanticSearch(List<SearchHit> keywordHits) {
    if (keywordHits.isEmpty) return true;

    final topScore = keywordHits.first.keywordScore;
    if (topScore <= 0) return true;

    final secondScore = keywordHits.length > 1
        ? keywordHits[1].keywordScore
        : 0.0;
    final margin = topScore - secondScore;

    if (topScore >= 2.4) return false;
    return topScore < 1.6 || margin < 0.25;
  }

  Future<void> installModelFromFile() async {
    if (isModelBusy) return;
    final stopwatch = Stopwatch()..start();
    AppLogger.info('setup.install_answer_engine_from_file.start');
    isModelBusy = true;
    error = null;
    status = 'Choose a local model file';
    notifyListeners();

    try {
      final result = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['task', 'litertlm', 'bin', 'tflite'],
        allowMultiple: false,
        withData: false,
      );
      final file = result?.files.single;
      final path = file?.path;
      if (path == null) {
        status = 'No model selected';
        AppLogger.warn('setup.install_answer_engine_from_file.cancelled', {
          'durationMs': stopwatch.elapsedMilliseconds,
        });
        return;
      }

      status = 'Installing ${file!.name}';
      AppLogger.info('setup.install_answer_engine_from_file.selected', {
        'fileName': file.name,
        'path': path,
      });
      notifyListeners();
      await aiService.installModelFromFile(
        path: path,
        modelType: selectedModel.modelType,
        fileType: _fileTypeFor(file.name),
      );
      hasLocalModel = aiService.hasActiveModel;
      status = 'Answer engine ready';
      AppLogger.info('setup.install_answer_engine_from_file.done', {
        'durationMs': stopwatch.elapsedMilliseconds,
        'hasAnswerEngine': hasLocalModel,
      });
    } catch (e, st) {
      AppLogger.error('setup.install_answer_engine_from_file.error', e, st, {
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      error = 'Model install failed: $e';
      status = 'Model not ready';
    } finally {
      isModelBusy = false;
      notifyListeners();
    }
  }

  Future<void> installSelectedModel() async {
    if (isModelBusy) return;
    final stopwatch = Stopwatch()..start();
    var lastLoggedProgress = -10;
    AppLogger.info('setup.install_answer_engine.start', {
      'label': selectedModel.label,
      'size': selectedModel.size,
    });
    isModelBusy = true;
    modelDownloadProgress = null;
    error = null;
    status = 'Checking answer engine';
    notifyListeners();

    try {
      if (await aiService.activateInstalledModel(selectedModel)) {
        hasLocalModel = true;
        modelDownloadProgress = null;
        status = 'Answer engine ready';
        AppLogger.info('setup.install_answer_engine.reused_installed', {
          'durationMs': stopwatch.elapsedMilliseconds,
          'hasAnswerEngine': hasLocalModel,
        });
        return;
      }

      modelDownloadProgress = 0;
      status = 'Downloading answer engine';
      notifyListeners();
      await aiService.installModelFromNetwork(
        model: selectedModel,
        onProgress: (progress) {
          modelDownloadProgress = progress;
          if (progress == 100 || progress - lastLoggedProgress >= 10) {
            lastLoggedProgress = progress;
            AppLogger.info('setup.install_answer_engine.progress', {
              'progress': progress,
            });
          }
          status = progress >= 100
              ? 'Installing answer engine'
              : 'Downloading answer engine ($progress%)';
          notifyListeners();
        },
      );
      hasLocalModel =
          await aiService.isModelInstalled(selectedModel) ||
          aiService.hasActiveModel;
      modelDownloadProgress = null;
      status = 'Answer engine ready';
      AppLogger.info('setup.install_answer_engine.done', {
        'durationMs': stopwatch.elapsedMilliseconds,
        'hasAnswerEngine': hasLocalModel,
      });
    } catch (e, st) {
      AppLogger.error('setup.install_answer_engine.error', e, st, {
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      error = 'Answer engine install failed: ${_describeFailure(e)}';
      status = 'Answer engine not ready';
      modelDownloadProgress = null;
    } finally {
      isModelBusy = false;
      notifyListeners();
    }
  }

  Future<void> installEmbeddingModel() async {
    if (isEmbeddingBusy || isModelBusy) return;
    final stopwatch = Stopwatch()..start();
    AppLogger.info('setup.install_semantic_search.start', {
      'supportsSemanticSearch': supportsEmbeddingModel,
      'supportedAbis': supportedAbis,
      'label': selectedEmbeddingModel.label,
      'size': selectedEmbeddingModel.size,
      'runtime': selectedEmbeddingModel.runtime.name,
    });
    if (!supportsEmbeddingModel) {
      error = supportedAbis.isEmpty
          ? 'Semantic search is not supported on this device.'
          : 'Semantic search needs an ARM64 Android device. Current ABI: ${supportedAbis.join(', ')}';
      status = 'Semantic retrieval not supported';
      AppLogger.warn('setup.install_semantic_search.unsupported', {
        'supportedAbis': supportedAbis,
      });
      notifyListeners();
      return;
    }

    var lastLoggedModelProgress = -10;
    var lastLoggedTokenizerProgress = -10;
    isEmbeddingBusy = true;
    embeddingModelDownloadProgress = null;
    embeddingTokenizerDownloadProgress = null;
    error = null;
    status = 'Checking semantic search';
    notifyListeners();

    try {
      if (await aiService.activateInstalledEmbedder(selectedEmbeddingModel)) {
        hasEmbeddingModel = true;
        embeddingModelDownloadProgress = null;
        embeddingTokenizerDownloadProgress = null;
        status = selectedEmbeddingModel.isBuiltIn
            ? 'Emulator semantic retrieval ready'
            : 'Semantic retrieval ready';
        AppLogger.info('setup.install_semantic_search.reused_installed', {
          'durationMs': stopwatch.elapsedMilliseconds,
          'hasSemanticSearch': hasEmbeddingModel,
          'runtime': selectedEmbeddingModel.runtime.name,
        });
      } else {
        embeddingModelDownloadProgress =
            selectedEmbeddingModel.requiresNetworkInstall ? 0 : null;
        embeddingTokenizerDownloadProgress =
            selectedEmbeddingModel.requiresNetworkInstall ? 0 : null;
        status = selectedEmbeddingModel.requiresNetworkInstall
            ? 'Downloading semantic search'
            : 'Preparing emulator semantic search';
        notifyListeners();
        await aiService.installEmbedderFromNetwork(
          model: selectedEmbeddingModel,
          onProgress: (modelProgress, tokenizerProgress) {
            if (selectedEmbeddingModel.requiresNetworkInstall) {
              embeddingModelDownloadProgress = modelProgress;
              embeddingTokenizerDownloadProgress = tokenizerProgress;
            }
            if (modelProgress == 100 ||
                tokenizerProgress == 100 ||
                modelProgress - lastLoggedModelProgress >= 10 ||
                tokenizerProgress - lastLoggedTokenizerProgress >= 10) {
              lastLoggedModelProgress = modelProgress;
              lastLoggedTokenizerProgress = tokenizerProgress;
              AppLogger.info('setup.install_semantic_search.progress', {
                'modelProgress': modelProgress,
                'tokenizerProgress': tokenizerProgress,
              });
            }
            status = selectedEmbeddingModel.requiresNetworkInstall
                ? 'Downloading semantic search '
                      '(model $modelProgress%, tokenizer $tokenizerProgress%)'
                : 'Preparing emulator semantic search';
            notifyListeners();
          },
        );
        hasEmbeddingModel = await aiService.isEmbeddingModelInstalled(
          selectedEmbeddingModel,
        );
        AppLogger.info('setup.install_semantic_search.download.done', {
          'hasSemanticSearch': hasEmbeddingModel,
          'runtime': selectedEmbeddingModel.runtime.name,
        });
      }

      final missing = await store.countChunksMissingEmbeddings();
      AppLogger.info('setup.install_semantic_search.backfill.check', {
        'missingEmbeddings': missing,
      });
      if (missing > 0) {
        status = 'Embedding indexed chunks (0/$missing)';
        notifyListeners();
        await store.backfillMissingEmbeddings(
          embedChunks: (chunks) => aiService.embedDocuments(
            chunks,
            maxInputChars: selectedEmbeddingModel.maxInputChars,
          ),
          embeddingModel: selectedEmbeddingModel.id,
          onProgress: (completed, total) {
            status = 'Embedding indexed chunks ($completed/$total)';
            AppLogger.info('setup.install_semantic_search.backfill.progress', {
              'completed': completed,
              'total': total,
            });
            notifyListeners();
          },
        );
      }

      await _refreshIndexState();
      status = stats.embeddedChunkCount > 0
          ? 'Hybrid retrieval ready'
          : 'Semantic retrieval ready';
      AppLogger.info('setup.install_semantic_search.done', {
        'durationMs': stopwatch.elapsedMilliseconds,
        'hasSemanticSearch': hasEmbeddingModel,
        'stats': _statsLog(),
      });
    } catch (e, st) {
      AppLogger.error('setup.install_semantic_search.error', e, st, {
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      error = 'Semantic model install failed: ${_describeFailure(e)}';
      status = 'Semantic retrieval not ready';
    } finally {
      embeddingModelDownloadProgress = null;
      embeddingTokenizerDownloadProgress = null;
      isEmbeddingBusy = false;
      notifyListeners();
    }
  }

  Future<void> clearIndex() async {
    AppLogger.warn('index.clear.start', {'stats': _statsLog()});
    await store.clear();
    await _refreshIndexState();
    status = 'Index cleared';
    AppLogger.warn('index.clear.done', {'stats': _statsLog()});
    notifyListeners();
  }

  Future<void> _refreshIndexState() async {
    documents = await store.listDocuments();
    stats = await store.stats();
    AppLogger.info('index.state.refreshed', {'stats': _statsLog()});
  }

  Future<List<List<double>>?> _embedChunksForIndexing(
    List<String> chunks,
    String name,
  ) async {
    try {
      status = 'Embedding $name';
      AppLogger.info('index.embed_chunks.start', {
        'document': name,
        'chunkCount': chunks.length,
        'chunkStats': AppLogger.textStats(chunks),
      });
      notifyListeners();
      final embeddings = await aiService.embedDocuments(
        chunks,
        maxInputChars: selectedEmbeddingModel.maxInputChars,
      );
      AppLogger.info('index.embed_chunks.done', {
        'document': name,
        'embeddingCount': embeddings.length,
        'dimension': embeddings.isEmpty ? 0 : embeddings.first.length,
      });
      return embeddings;
    } catch (e, st) {
      AppLogger.error('index.embed_chunks.error', e, st, {'document': name});
      error = 'Indexed $name without semantic vectors: ${_describeFailure(e)}';
      return null;
    }
  }

  ModelFileType _fileTypeFor(String fileName) {
    final lower = fileName.toLowerCase();
    if (lower.endsWith('.bin') || lower.endsWith('.tflite')) {
      return ModelFileType.binary;
    }
    return ModelFileType.task;
  }

  Future<void> disposeApp() async {
    AppLogger.info('app.dispose.start');
    await aiService.dispose();
    await store.close();
    AppLogger.info('app.dispose.done');
  }

  String _describeFailure(Object error) {
    final message = error is PlatformException
        ? () {
            final details = error.details == null
                ? ''
                : ' Details: ${error.details}';
            return '${error.code}: ${error.message ?? error.toString()}$details';
          }()
        : error.toString();
    return _hideModelNames(message);
  }

  String _hideModelNames(String message) {
    var sanitized = message;
    for (final pattern in [
      RegExp(r'https://huggingface\.co/\S+', caseSensitive: false),
      RegExp(r'\S*Qwen\S*', caseSensitive: false),
      RegExp(r'\S*Gecko\S*', caseSensitive: false),
    ]) {
      sanitized = sanitized.replaceAllMapped(pattern, (match) {
        final text = match.group(0) ?? '';
        if (text.toLowerCase().contains('gecko')) {
          return 'semantic-search-file';
        }
        if (text.toLowerCase().contains('qwen')) {
          return 'answer-engine-file';
        }
        return 'remote-model-file';
      });
    }
    return sanitized;
  }

  Map<String, Object?> _statsLog() {
    return {
      'documents': stats.documentCount,
      'chunks': stats.chunkCount,
      'embeddedChunks': stats.embeddedChunkCount,
      'totalBytes': stats.totalBytes,
    };
  }

  List<Map<String, Object?>> _candidateSummaries(
    List<ImportCandidate> candidates,
  ) {
    return candidates.take(12).map(_candidateSummary).toList(growable: false);
  }

  Map<String, Object?> _candidateSummary(ImportCandidate candidate) {
    return {
      'name': candidate.name,
      'sourcePath': candidate.sourcePath,
      'hasPath': candidate.path != null,
      'hasBytes': candidate.bytes != null,
      'hasReadStream': candidate.readStream != null,
      if (candidate.bytes != null) 'bytes': candidate.bytes!.length,
    };
  }

  List<Map<String, Object?>> _hitSummaries(List<SearchHit> hits) {
    return [
      for (var i = 0; i < hits.length && i < 8; i++) _hitSummary(hits[i], i),
    ];
  }

  Map<String, Object?> _hitSummary(SearchHit hit, int index) {
    return {
      'rank': index + 1,
      'document': hit.document.name,
      'chunkIndex': hit.chunk.index,
      'score': AppLogger.score(hit.score),
      'keywordScore': AppLogger.score(hit.keywordScore),
      'vectorScore': AppLogger.score(hit.vectorScore),
      'chunkChars': hit.chunk.content.length,
      'contextChars': hit.contextText.length,
      'preview': AppLogger.preview(hit.contextText, 140),
    };
  }
}
