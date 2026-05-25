import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:vaultiq/src/model_catalog.dart';
import 'package:vaultiq/src/models.dart';
import 'package:vaultiq/src/services/document_store.dart';
import 'package:vaultiq/src/services/local_ai_service.dart';
import 'package:vaultiq/src/services/text_chunker.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('150-case RAG quality evaluation', (tester) async {
    final generated = _buildDataset();
    final appDir = await getApplicationDocumentsDirectory();
    final corpusFile = File('${appDir.path}/local_doc_qa_eval_corpus.md');
    await corpusFile.writeAsString(generated.corpus);

    final chunker = TextChunker();
    final chunks = chunker.chunk(generated.corpus);
    final chunkStats = _scoreChunks(generated.records, chunks);

    final store = DocumentStore(chunker: chunker);
    final ai = LocalAiService();
    await store.open();
    await ai.initialize();
    await store.clear();

    await store.upsertDocument(
      ExtractedDocument(
        name: 'local_doc_qa_eval_corpus.md',
        sourcePath: corpusFile.path,
        bytes: utf8.encode(generated.corpus).length,
        text: generated.corpus,
      ),
    );

    if (!ai.hasActiveModel) {
      await ai.installModelFromNetwork(
        model: DownloadableModel.qwen25OnePointFiveB,
        onProgress: (progress) {
          // ignore: avoid_print
          print('Answer engine install progress: $progress%');
        },
      );
    }

    final config = DownloadableModel.qwen25OnePointFiveB.inferenceConfig
        .copyWith(
          temperature: 0.1,
          topK: 20,
          topP: 0.85,
          reservedOutputTokens: 220,
          maxPromptHits: 3,
          maxContextChars: 1600,
          maxExcerptChars: 520,
          maxQuestionChars: 320,
        );

    final results = <_EvalResult>[];
    final stopwatch = Stopwatch()..start();
    for (var i = 0; i < generated.cases.length; i++) {
      final testCase = generated.cases[i];
      final caseWatch = Stopwatch()..start();
      final hits = await store.search(testCase.question, topK: 8);
      final answer = await ai.answer(
        question: testCase.question,
        hits: hits,
        config: config,
      );
      caseWatch.stop();

      final result = _scoreCase(
        index: i + 1,
        testCase: testCase,
        hits: hits,
        answer: answer,
        elapsedMs: caseWatch.elapsedMilliseconds,
      );
      results.add(result);

      if ((i + 1) % 10 == 0) {
        // ignore: avoid_print
        print(
          'RAG eval progress: ${i + 1}/${generated.cases.length}, '
          'running average ${_average(results.map((r) => r.totalScore)).toStringAsFixed(2)}',
        );
      }
    }
    stopwatch.stop();

    final report = _buildReport(
      chunkStats: chunkStats,
      results: results,
      elapsedMs: stopwatch.elapsedMilliseconds,
      corpusPath: corpusFile.path,
    );
    final reportFile = File('${appDir.path}/rag_quality_eval_report.md');
    await reportFile.writeAsString(report);
    final jsonFile = File('${appDir.path}/rag_quality_eval_results.json');
    final jsonReport = const JsonEncoder.withIndent('  ').convert({
      'corpusPath': corpusFile.path,
      'reportPath': reportFile.path,
      'chunkStats': chunkStats.toJson(),
      'summary': _summaryJson(results, stopwatch.elapsedMilliseconds),
      'results': [for (final result in results) result.toJson()],
    });
    await jsonFile.writeAsString(jsonReport);

    // ignore: avoid_print
    print('RAG_EVAL_REPORT_PATH=${reportFile.path}');
    // ignore: avoid_print
    print('RAG_EVAL_JSON_PATH=${jsonFile.path}');
    _printBase64Artifact('RAG_EVAL_REPORT_MD', report);
    _printBase64Artifact('RAG_EVAL_RESULTS_JSON', jsonReport);

    await ai.dispose();
    await store.close();

    expect(results, hasLength(150));
  }, timeout: const Timeout(Duration(hours: 4)));
}

_GeneratedDataset _buildDataset() {
  final owners = [
    'Neon Platform',
    'Blue Harbor',
    'Quartz Finance',
    'Cedar Security',
    'Orbit Support',
    'Lumen Data',
    'Aster Compliance',
    'Northstar Ops',
  ];
  final approvals = [
    'manager approval',
    'finance director approval',
    'security review',
    'legal review',
    'data steward approval',
    'incident commander approval',
  ];
  final regions = [
    'Bengaluru',
    'Seattle',
    'Dublin',
    'Singapore',
    'Toronto',
    'Berlin',
  ];
  final codenames = [
    'Atlas Relay',
    'Beacon Ledger',
    'Cobalt Bridge',
    'Delta Archive',
    'Ember Queue',
    'Frost Gateway',
    'Granite Sync',
    'Helio Vault',
    'Ion Parser',
    'Juniper Mesh',
  ];

  final records = <_FactRecord>[];
  for (var i = 1; i <= 60; i++) {
    final id = 'CASE-${i.toString().padLeft(3, '0')}';
    records.add(
      _FactRecord(
        id: id,
        codename: '${codenames[i % codenames.length]} $i',
        owner: owners[i % owners.length],
        approval: approvals[i % approvals.length],
        region: regions[i % regions.length],
        deadline:
            '2026-${((i % 9) + 1).toString().padLeft(2, '0')}-${((i * 3) % 27 + 1).toString().padLeft(2, '0')}',
        budget: 1200 + (i * 47),
        retentionDays: 14 + (i % 21),
        severity: ['low', 'medium', 'high', 'critical'][i % 4],
      ),
    );
  }

  final corpus = StringBuffer()
    ..writeln('# Local Doc QA Evaluation Corpus')
    ..writeln()
    ..writeln(
      'This file is a synthetic evaluation corpus for local document question answering. '
      'Every record below is authoritative for the tests. If a question asks for a record or field '
      'that is not listed, the correct behavior is to say the local context is missing.',
    )
    ..writeln();

  for (final record in records) {
    corpus
      ..writeln('## ${record.id}: ${record.codename}')
      ..writeln('Record ID: ${record.id}.')
      ..writeln('Project codename: ${record.codename}.')
      ..writeln('Owner team: ${record.owner}.')
      ..writeln('Approval path: ${record.approval}.')
      ..writeln('Primary region: ${record.region}.')
      ..writeln('Deadline: ${record.deadline}.')
      ..writeln('Budget cap: \$${record.budget}.')
      ..writeln('Retention window: ${record.retentionDays} days.')
      ..writeln('Severity class: ${record.severity}.')
      ..writeln(
        'Operational note: when answering questions about ${record.id}, use the exact owner, '
        'approval path, deadline, region, budget, retention window, and severity values written in this section.',
      )
      ..writeln();
  }

  final cases = <_EvalCase>[];
  for (var i = 0; i < 40; i++) {
    final record = records[i];
    cases.add(
      _EvalCase(
        id: 'OWNER-${record.id}',
        category: 'single-owner',
        question:
            'For ${record.id}, answer with only the owner team and cite the file.',
        expectedRecordId: record.id,
        requiredTerms: [record.owner],
      ),
    );
  }
  for (var i = 10; i < 50; i++) {
    final record = records[i];
    cases.add(
      _EvalCase(
        id: 'DEADLINE-${record.id}',
        category: 'single-deadline',
        question:
            'What is the exact deadline for ${record.id}? Answer briefly and cite the file.',
        expectedRecordId: record.id,
        requiredTerms: [record.deadline],
      ),
    );
  }
  for (var i = 20; i < 50; i++) {
    final record = records[i];
    cases.add(
      _EvalCase(
        id: 'BUDGET-${record.id}',
        category: 'numeric-budget',
        question:
            'What budget cap is listed for ${record.id}? Answer with the amount only if possible.',
        expectedRecordId: record.id,
        requiredTerms: [record.budget.toString()],
      ),
    );
  }
  for (var i = 0; i < 25; i++) {
    final record = records[(i * 2) % records.length];
    cases.add(
      _EvalCase(
        id: 'MULTI-${record.id}',
        category: 'multi-field',
        question:
            'For ${record.id}, provide the approval path and primary region in one concise sentence.',
        expectedRecordId: record.id,
        requiredTerms: [record.approval, record.region],
      ),
    );
  }
  for (var i = 0; i < 10; i++) {
    final record = records[(i * 5 + 3) % records.length];
    cases.add(
      _EvalCase(
        id: 'RETENTION-${record.id}',
        category: 'numeric-retention',
        question:
            'How many days is the retention window for ${record.id}? Answer briefly.',
        expectedRecordId: record.id,
        requiredTerms: [record.retentionDays.toString(), 'days'],
      ),
    );
  }
  for (var i = 0; i < 5; i++) {
    final missingId = 'CASE-${(90 + i).toString().padLeft(3, '0')}';
    cases.add(
      _EvalCase(
        id: 'MISSING-$missingId',
        category: 'unanswerable',
        question:
            'What is the owner team for $missingId? If the local file does not say, state that it is missing.',
        expectedRecordId: missingId,
        requiredTerms: ['missing'],
        expectsMissing: true,
      ),
    );
  }

  return _GeneratedDataset(
    corpus: corpus.toString(),
    records: records,
    cases: cases,
  );
}

_EvalResult _scoreCase({
  required int index,
  required _EvalCase testCase,
  required List<SearchHit> hits,
  required String answer,
  required int elapsedMs,
}) {
  final normalizedAnswer = _normalize(answer);
  final expectedTerms = testCase.requiredTerms.map(_normalize).toList();

  final retrievalHit = testCase.expectsMissing
      ? hits.every(
          (hit) => !hit.contextText.contains(testCase.expectedRecordId),
        )
      : hits.any((hit) => hit.contextText.contains(testCase.expectedRecordId));
  final retrievalScore = retrievalHit ? 1.0 : 0.0;

  final answerMatches = testCase.expectsMissing
      ? _missingSignals.any(normalizedAnswer.contains)
      : expectedTerms.where(normalizedAnswer.contains).length /
            expectedTerms.length;
  final answerScore = answerMatches is bool
      ? (answerMatches ? 1.0 : 0.0)
      : (answerMatches as double).clamp(0.0, 1.0);

  final citationScore =
      answer.contains('[local_doc_qa_eval_corpus.md]') ||
          answer.contains('local_doc_qa_eval_corpus.md') ||
          RegExp(r'\[[^\]]+\]').hasMatch(answer)
      ? 1.0
      : 0.0;
  final totalScore =
      (answerScore * 0.65) + (retrievalScore * 0.25) + (citationScore * 0.10);

  return _EvalResult(
    index: index,
    id: testCase.id,
    category: testCase.category,
    question: testCase.question,
    expectedRecordId: testCase.expectedRecordId,
    requiredTerms: testCase.requiredTerms,
    answer: answer,
    retrievalScore: retrievalScore,
    answerScore: answerScore,
    citationScore: citationScore,
    totalScore: totalScore,
    topSources: [
      for (final hit in hits.take(3))
        '${hit.document.name}#${hit.chunk.index + 1}',
    ],
    elapsedMs: elapsedMs,
  );
}

_ChunkStats _scoreChunks(List<_FactRecord> records, List<String> chunks) {
  final lengths = chunks.map((chunk) => chunk.length).toList(growable: false);
  final completeRecords = records.where((record) {
    return chunks.any(
      (chunk) =>
          chunk.contains(record.id) &&
          chunk.contains(record.owner) &&
          chunk.contains(record.deadline) &&
          chunk.contains(record.budget.toString()),
    );
  }).length;

  return _ChunkStats(
    chunkCount: chunks.length,
    minChars: lengths.reduce(math.min),
    maxChars: lengths.reduce(math.max),
    averageChars: lengths.reduce((a, b) => a + b) / lengths.length,
    completeRecordCoverage: completeRecords / records.length,
  );
}

String _buildReport({
  required _ChunkStats chunkStats,
  required List<_EvalResult> results,
  required int elapsedMs,
  required String corpusPath,
}) {
  final byCategory = <String, List<_EvalResult>>{};
  for (final result in results) {
    byCategory.putIfAbsent(result.category, () => []).add(result);
  }

  final buffer = StringBuffer()
    ..writeln('# Local Doc QA RAG Evaluation Report')
    ..writeln()
    ..writeln('- Test cases: ${results.length}')
    ..writeln('- Corpus path on emulator: `$corpusPath`')
    ..writeln('- Total runtime: ${(elapsedMs / 1000).toStringAsFixed(1)}s')
    ..writeln(
      '- Mean total score: ${_average(results.map((r) => r.totalScore)).toStringAsFixed(3)}',
    )
    ..writeln(
      '- Mean answer score: ${_average(results.map((r) => r.answerScore)).toStringAsFixed(3)}',
    )
    ..writeln(
      '- Retrieval hit rate: ${_average(results.map((r) => r.retrievalScore)).toStringAsFixed(3)}',
    )
    ..writeln(
      '- Citation rate: ${_average(results.map((r) => r.citationScore)).toStringAsFixed(3)}',
    )
    ..writeln()
    ..writeln('## Chunking')
    ..writeln()
    ..writeln('- Chunks: ${chunkStats.chunkCount}')
    ..writeln('- Min chars: ${chunkStats.minChars}')
    ..writeln('- Max chars: ${chunkStats.maxChars}')
    ..writeln('- Average chars: ${chunkStats.averageChars.toStringAsFixed(1)}')
    ..writeln(
      '- Complete record coverage: ${(chunkStats.completeRecordCoverage * 100).toStringAsFixed(1)}%',
    )
    ..writeln()
    ..writeln('## Scores By Category')
    ..writeln()
    ..writeln('| Category | Cases | Total | Answer | Retrieval | Citation |')
    ..writeln('|---|---:|---:|---:|---:|---:|');

  for (final entry in byCategory.entries) {
    buffer.writeln(
      '| ${entry.key} | ${entry.value.length} | '
      '${_average(entry.value.map((r) => r.totalScore)).toStringAsFixed(3)} | '
      '${_average(entry.value.map((r) => r.answerScore)).toStringAsFixed(3)} | '
      '${_average(entry.value.map((r) => r.retrievalScore)).toStringAsFixed(3)} | '
      '${_average(entry.value.map((r) => r.citationScore)).toStringAsFixed(3)} |',
    );
  }

  buffer
    ..writeln()
    ..writeln('## Per-Case Results')
    ..writeln()
    ..writeln(
      '| # | ID | Category | Score | Retrieval | Answer | Citation | Latency ms | Expected | Output |',
    )
    ..writeln('|---:|---|---|---:|---:|---:|---:|---:|---|---|');

  for (final result in results) {
    buffer.writeln(
      '| ${result.index} | ${result.id} | ${result.category} | '
      '${result.totalScore.toStringAsFixed(2)} | '
      '${result.retrievalScore.toStringAsFixed(0)} | '
      '${result.answerScore.toStringAsFixed(2)} | '
      '${result.citationScore.toStringAsFixed(0)} | '
      '${result.elapsedMs} | '
      '${_escapeTable(result.requiredTerms.join('; '))} | '
      '${_escapeTable(_clip(result.answer, 180))} |',
    );
  }

  return buffer.toString();
}

Map<String, Object?> _summaryJson(List<_EvalResult> results, int elapsedMs) {
  return {
    'caseCount': results.length,
    'elapsedMs': elapsedMs,
    'meanTotalScore': _average(results.map((r) => r.totalScore)),
    'meanAnswerScore': _average(results.map((r) => r.answerScore)),
    'retrievalHitRate': _average(results.map((r) => r.retrievalScore)),
    'citationRate': _average(results.map((r) => r.citationScore)),
  };
}

double _average(Iterable<double> values) {
  final list = values.toList(growable: false);
  if (list.isEmpty) return 0;
  return list.reduce((a, b) => a + b) / list.length;
}

String _normalize(String value) {
  return value.toLowerCase().replaceAll(RegExp(r'\s+'), ' ').trim();
}

String _clip(String value, int maxChars) {
  final normalized = value.replaceAll(RegExp(r'\s+'), ' ').trim();
  if (normalized.length <= maxChars) return normalized;
  return '${normalized.substring(0, maxChars - 3)}...';
}

String _escapeTable(String value) {
  return value.replaceAll('|', r'\|').replaceAll('\n', ' ');
}

void _printBase64Artifact(String name, String content) {
  final encoded = base64Encode(utf8.encode(content));
  const chunkSize = 900;
  // ignore: avoid_print
  print('${name}_BASE64_BEGIN');
  for (var i = 0; i < encoded.length; i += chunkSize) {
    final end = math.min(i + chunkSize, encoded.length);
    // ignore: avoid_print
    print('$name:${encoded.substring(i, end)}');
  }
  // ignore: avoid_print
  print('${name}_BASE64_END');
}

const _missingSignals = [
  'missing',
  'not listed',
  'not available',
  'could not find',
  'no relevant',
  'does not say',
  'not in the local',
];

class _GeneratedDataset {
  const _GeneratedDataset({
    required this.corpus,
    required this.records,
    required this.cases,
  });

  final String corpus;
  final List<_FactRecord> records;
  final List<_EvalCase> cases;
}

class _FactRecord {
  const _FactRecord({
    required this.id,
    required this.codename,
    required this.owner,
    required this.approval,
    required this.region,
    required this.deadline,
    required this.budget,
    required this.retentionDays,
    required this.severity,
  });

  final String id;
  final String codename;
  final String owner;
  final String approval;
  final String region;
  final String deadline;
  final int budget;
  final int retentionDays;
  final String severity;
}

class _EvalCase {
  const _EvalCase({
    required this.id,
    required this.category,
    required this.question,
    required this.expectedRecordId,
    required this.requiredTerms,
    this.expectsMissing = false,
  });

  final String id;
  final String category;
  final String question;
  final String expectedRecordId;
  final List<String> requiredTerms;
  final bool expectsMissing;
}

class _EvalResult {
  const _EvalResult({
    required this.index,
    required this.id,
    required this.category,
    required this.question,
    required this.expectedRecordId,
    required this.requiredTerms,
    required this.answer,
    required this.retrievalScore,
    required this.answerScore,
    required this.citationScore,
    required this.totalScore,
    required this.topSources,
    required this.elapsedMs,
  });

  final int index;
  final String id;
  final String category;
  final String question;
  final String expectedRecordId;
  final List<String> requiredTerms;
  final String answer;
  final double retrievalScore;
  final double answerScore;
  final double citationScore;
  final double totalScore;
  final List<String> topSources;
  final int elapsedMs;

  Map<String, Object?> toJson() {
    return {
      'index': index,
      'id': id,
      'category': category,
      'question': question,
      'expectedRecordId': expectedRecordId,
      'requiredTerms': requiredTerms,
      'answer': answer,
      'retrievalScore': retrievalScore,
      'answerScore': answerScore,
      'citationScore': citationScore,
      'totalScore': totalScore,
      'topSources': topSources,
      'elapsedMs': elapsedMs,
    };
  }
}

class _ChunkStats {
  const _ChunkStats({
    required this.chunkCount,
    required this.minChars,
    required this.maxChars,
    required this.averageChars,
    required this.completeRecordCoverage,
  });

  final int chunkCount;
  final int minChars;
  final int maxChars;
  final double averageChars;
  final double completeRecordCoverage;

  Map<String, Object?> toJson() {
    return {
      'chunkCount': chunkCount,
      'minChars': minChars,
      'maxChars': maxChars,
      'averageChars': averageChars,
      'completeRecordCoverage': completeRecordCoverage,
    };
  }
}

extension on LocalInferenceConfig {
  LocalInferenceConfig copyWith({
    int? maxTokens,
    double? temperature,
    int? topK,
    double? topP,
    int? reservedOutputTokens,
    int? maxPromptHits,
    int? maxContextChars,
    int? maxExcerptChars,
    int? maxQuestionChars,
  }) {
    return LocalInferenceConfig(
      preferredBackend: preferredBackend,
      maxTokens: maxTokens ?? this.maxTokens,
      temperature: temperature ?? this.temperature,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      reservedOutputTokens: reservedOutputTokens ?? this.reservedOutputTokens,
      maxPromptHits: maxPromptHits ?? this.maxPromptHits,
      maxContextChars: maxContextChars ?? this.maxContextChars,
      maxExcerptChars: maxExcerptChars ?? this.maxExcerptChars,
      maxQuestionChars: maxQuestionChars ?? this.maxQuestionChars,
    );
  }
}
