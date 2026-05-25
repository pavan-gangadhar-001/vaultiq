import 'dart:math' as math;

import 'package:collection/collection.dart';

import '../models.dart';
import 'app_logger.dart';

class RetrievalCandidate {
  const RetrievalCandidate({
    required this.document,
    required this.chunk,
    this.keywordScore,
  });

  final IndexedDocument document;
  final DocumentChunk chunk;
  final double? keywordScore;
}

class RetrievalRanker {
  const RetrievalRanker({
    this.bm25K1 = 1.5,
    this.bm25B = 0.75,
    this.rrfK = 60,
    this.candidateMultiplier = 8,
    this.maxChunksPerDocument = 2,
    this.mmrLambda = 0.72,
  });

  final double bm25K1;
  final double bm25B;
  final double rrfK;
  final int candidateMultiplier;
  final int maxChunksPerDocument;
  final double mmrLambda;

  List<SearchHit> rank({
    required String query,
    required List<RetrievalCandidate> candidates,
    List<double>? queryEmbedding,
    int topK = 5,
  }) {
    final stopwatch = Stopwatch()..start();
    AppLogger.info('ranker.rank.start', {
      'queryChars': query.length,
      'queryPreview': AppLogger.preview(query),
      'candidateCount': candidates.length,
      'candidatesWithEmbedding': candidates
          .where((candidate) => candidate.chunk.embedding != null)
          .length,
      'queryEmbeddingDim': queryEmbedding?.length ?? 0,
      'topK': topK,
      'candidateMultiplier': candidateMultiplier,
      'maxChunksPerDocument': maxChunksPerDocument,
      'mmrLambda': mmrLambda,
    });
    if (topK <= 0 || candidates.isEmpty) {
      AppLogger.warn('ranker.rank.empty_input', {
        'topK': topK,
        'candidateCount': candidates.length,
      });
      return const [];
    }

    final queryIntent = _QueryIntent.fromQuery(query);
    if (queryIntent.queryTerms.isEmpty && queryEmbedding == null) {
      AppLogger.warn('ranker.rank.no_signals', {
        'queryPreview': AppLogger.preview(query),
      });
      return const [];
    }

    final poolSize = math.max(topK, topK * candidateMultiplier);
    final keywordHits = queryIntent.queryTerms.isEmpty
        ? <SearchHit>[]
        : _rankBm25(
            queryTerms: queryIntent.queryTerms.toSet(),
            candidates: candidates,
          ).take(poolSize).toList(growable: false);
    final vectorHits = queryEmbedding == null
        ? <SearchHit>[]
        : _rankVectors(
            queryEmbedding: queryEmbedding,
            candidates: candidates,
          ).take(poolSize).toList(growable: false);

    final firstStageRanked = vectorHits.isEmpty
        ? keywordHits
        : _fuseRankings(keywordHits: keywordHits, vectorHits: vectorHits);
    final reranked = _rerank(queryIntent: queryIntent, hits: firstStageRanked);

    final diversified = _diversify(reranked, topK).toList(growable: false);
    AppLogger.info('ranker.rank.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'queryTerms': queryIntent.queryTerms,
      'queryIdentifiers': queryIntent.identifiers.toList(),
      'queryNumbers': queryIntent.numbers.toList(),
      'poolSize': poolSize,
      'keywordHitCount': keywordHits.length,
      'vectorHitCount': vectorHits.length,
      'firstStageCount': firstStageRanked.length,
      'rerankedCount': reranked.length,
      'resultCount': diversified.length,
      'results': _hitSummaries(diversified),
    });
    return diversified;
  }

  List<SearchHit> _rankBm25({
    required Set<String> queryTerms,
    required List<RetrievalCandidate> candidates,
  }) {
    final indexedHits = candidates
        .where((candidate) => (candidate.keywordScore ?? 0) > 0)
        .map(
          (candidate) => SearchHit(
            document: candidate.document,
            chunk: candidate.chunk,
            score: candidate.keywordScore!,
            keywordScore: candidate.keywordScore!,
          ),
        )
        .toList(growable: false);
    if (indexedHits.isNotEmpty) {
      indexedHits.sort((a, b) => b.score.compareTo(a.score));
      return indexedHits;
    }

    final tokenized = candidates
        .map((candidate) {
          final terms = _tokenize(candidate.chunk.content);
          final titleTerms = _tokenize(candidate.document.name).toSet();
          return _TokenizedCandidate(
            candidate: candidate,
            terms: terms,
            titleTerms: titleTerms,
          );
        })
        .where((candidate) => candidate.terms.isNotEmpty)
        .toList(growable: false);
    if (tokenized.isEmpty) return const [];

    final docCount = tokenized.length;
    final averageLength =
        tokenized.fold<int>(0, (sum, candidate) => sum + candidate.length) /
        docCount;
    final documentFrequencies = <String, int>{};

    for (final term in queryTerms) {
      documentFrequencies[term] = tokenized
          .where((candidate) => candidate.uniqueTerms.contains(term))
          .length;
    }

    final hits = <SearchHit>[];
    for (final candidate in tokenized) {
      var score = 0.0;
      for (final term in queryTerms) {
        final frequency = candidate.frequencies[term] ?? 0;
        final inTitle = candidate.titleTerms.contains(term);
        if (frequency == 0 && !inTitle) continue;

        final df = documentFrequencies[term] ?? 0;
        final idf = math.log(1 + (docCount - df + 0.5) / (df + 0.5));
        if (frequency > 0) {
          final numerator = frequency * (bm25K1 + 1);
          final denominator =
              frequency +
              bm25K1 * (1 - bm25B + bm25B * candidate.length / averageLength);
          score += idf * numerator / denominator;
        }
        if (inTitle) score += idf * 1.25;
      }

      final coverage =
          queryTerms
              .where(
                (term) =>
                    candidate.frequencies.containsKey(term) ||
                    candidate.titleTerms.contains(term),
              )
              .length /
          queryTerms.length;
      score += coverage * 0.35;

      if (score > 0 && score.isFinite) {
        hits.add(
          SearchHit(
            document: candidate.candidate.document,
            chunk: candidate.candidate.chunk,
            score: score,
            keywordScore: score,
          ),
        );
      }
    }

    hits.sort((a, b) => b.score.compareTo(a.score));
    return hits;
  }

  List<SearchHit> _rankVectors({
    required List<double> queryEmbedding,
    required List<RetrievalCandidate> candidates,
  }) {
    if (queryEmbedding.isEmpty) return const [];
    final queryNorm = _vectorNorm(queryEmbedding);
    if (queryNorm == 0) return const [];

    final hits = <SearchHit>[];
    for (final candidate in candidates) {
      final embedding = candidate.chunk.embedding;
      if (embedding == null || embedding.length != queryEmbedding.length) {
        continue;
      }

      final score = _cosineSimilarity(
        queryEmbedding,
        embedding,
        normA: queryNorm,
      );
      if (score > 0 && score.isFinite) {
        hits.add(
          SearchHit(
            document: candidate.document,
            chunk: candidate.chunk,
            score: score,
            vectorScore: score,
          ),
        );
      }
    }

    hits.sort((a, b) => b.score.compareTo(a.score));
    return hits;
  }

  List<SearchHit> _fuseRankings({
    required List<SearchHit> keywordHits,
    required List<SearchHit> vectorHits,
  }) {
    final entries = <String, _FusionEntry>{};

    void addHits(List<SearchHit> hits, _RetrievalSignal signal) {
      for (var i = 0; i < hits.length; i++) {
        final hit = hits[i];
        final entry = entries.putIfAbsent(
          hit.chunk.id,
          () => _FusionEntry(document: hit.document, chunk: hit.chunk),
        );
        entry.score += 1 / (rrfK + i + 1);
        switch (signal) {
          case _RetrievalSignal.keyword:
            entry.keywordScore = math.max(entry.keywordScore, hit.keywordScore);
          case _RetrievalSignal.vector:
            entry.vectorScore = math.max(entry.vectorScore, hit.vectorScore);
        }
      }
    }

    addHits(keywordHits, _RetrievalSignal.keyword);
    addHits(vectorHits, _RetrievalSignal.vector);

    final fused = entries.values
        .map(
          (entry) => SearchHit(
            document: entry.document,
            chunk: entry.chunk,
            score: entry.score,
            keywordScore: entry.keywordScore,
            vectorScore: entry.vectorScore,
          ),
        )
        .toList(growable: false);
    fused.sort((a, b) => b.score.compareTo(a.score));
    return fused;
  }

  List<SearchHit> _rerank({
    required _QueryIntent queryIntent,
    required List<SearchHit> hits,
  }) {
    if (hits.length <= 1) return hits;

    final maxFirstStageScore = _maxScore(hits.map((hit) => hit.score));
    final maxKeywordScore = _maxScore(hits.map((hit) => hit.keywordScore));
    final maxVectorScore = _maxScore(hits.map((hit) => hit.vectorScore));
    final reranked = <_RerankedHit>[];

    for (final hit in hits) {
      final evidence = '${hit.document.name}\n${hit.chunk.content}';
      final evidenceTerms = _tokenize(evidence);
      final evidenceTermSet = evidenceTerms.toSet();
      final normalizedEvidence = _normalizeForPhrase(evidence);
      final titleTerms = _tokenize(hit.document.name).toSet();

      final firstStageScore = _normalizeScore(hit.score, maxFirstStageScore);
      final keywordScore = _normalizeScore(hit.keywordScore, maxKeywordScore);
      final vectorScore = _normalizeScore(hit.vectorScore, maxVectorScore);
      final coverageScore = _coverageScore(
        queryIntent.queryTerms,
        evidenceTermSet,
      );
      final phraseScore = _phraseScore(queryIntent.phrases, normalizedEvidence);
      final identifierScore = _exactSetScore(
        queryIntent.identifiers,
        _extractIdentifiers(evidence),
      );
      final numberScore = _exactSetScore(
        queryIntent.numbers,
        _extractNumbers(evidence),
      );
      final titleScore = _coverageScore(queryIntent.queryTerms, titleTerms);
      final proximityScore = _proximityScore(
        queryIntent.queryTerms,
        evidenceTerms,
      );
      final fieldCueScore = _fieldCueScore(
        queryIntent.fieldCues,
        normalizedEvidence,
      );

      var rerankScore =
          firstStageScore * 0.24 +
          keywordScore * 0.20 +
          vectorScore * 0.14 +
          coverageScore * 1.25 +
          phraseScore * 0.80 +
          identifierScore * 2.30 +
          numberScore * 2.35 +
          titleScore * 0.55 +
          proximityScore * 0.75 +
          fieldCueScore * 0.25;

      if (queryIntent.identifiers.isNotEmpty && identifierScore == 0) {
        rerankScore -= 1.35;
      }
      if (queryIntent.numbers.isNotEmpty && numberScore == 0) {
        rerankScore -= 1.40;
      }

      if (rerankScore.isFinite) {
        reranked.add(
          _RerankedHit(
            hit: SearchHit(
              document: hit.document,
              chunk: hit.chunk,
              score: math.max(rerankScore, 0.000001),
              keywordScore: hit.keywordScore,
              vectorScore: hit.vectorScore,
            ),
            firstStageScore: firstStageScore,
            coverageScore: coverageScore,
            phraseScore: phraseScore,
            identifierScore: identifierScore,
            numberScore: numberScore,
            titleScore: titleScore,
            proximityScore: proximityScore,
            fieldCueScore: fieldCueScore,
          ),
        );
      }
    }

    reranked.sort((a, b) => b.hit.score.compareTo(a.hit.score));
    AppLogger.info('ranker.rerank.done', {
      'inputCount': hits.length,
      'outputCount': reranked.length,
      'identifiers': queryIntent.identifiers.toList(),
      'numbers': queryIntent.numbers.toList(),
      'phrases': queryIntent.phrases.take(8).toList(),
      'fieldCues': queryIntent.fieldCues.toList(),
      'results': [
        for (var i = 0; i < reranked.length && i < 8; i++)
          reranked[i].summary(i),
      ],
    });
    return reranked.map((entry) => entry.hit).toList(growable: false);
  }

  Iterable<SearchHit> _diversify(List<SearchHit> hits, int topK) sync* {
    if (hits.length <= 1) {
      yield* hits.take(topK);
      return;
    }

    final selected = <SearchHit>[];
    final selectedIds = <String>{};
    final perDocument = <String, int>{};

    for (final cap in [maxChunksPerDocument, topK]) {
      while (selected.length < topK) {
        final next = _nextMmrHit(
          hits,
          selected: selected,
          selectedIds: selectedIds,
          perDocument: perDocument,
          perDocumentCap: cap,
        );
        if (next == null) break;

        selected.add(next);
        selectedIds.add(next.chunk.id);
        perDocument[next.document.id] =
            (perDocument[next.document.id] ?? 0) + 1;
        yield next;
      }
      if (selected.length >= topK) return;
    }
  }

  SearchHit? _nextMmrHit(
    List<SearchHit> hits, {
    required List<SearchHit> selected,
    required Set<String> selectedIds,
    required Map<String, int> perDocument,
    required int perDocumentCap,
  }) {
    final maxScore = hits.first.score <= 0 ? 1.0 : hits.first.score;
    SearchHit? bestHit;
    var bestScore = double.negativeInfinity;

    for (final hit in hits) {
      if (selectedIds.contains(hit.chunk.id)) continue;

      final count = perDocument[hit.document.id] ?? 0;
      if (count >= perDocumentCap) continue;

      final relevance = (hit.score / maxScore).clamp(0.0, 1.0);
      final redundancy = selected.isEmpty
          ? 0.0
          : selected
                .map((selectedHit) => _hitSimilarity(hit, selectedHit))
                .fold<double>(0, math.max);
      final score = mmrLambda * relevance - (1 - mmrLambda) * redundancy;
      if (score > bestScore) {
        bestScore = score;
        bestHit = hit;
      }
    }

    return bestHit;
  }

  double _hitSimilarity(SearchHit a, SearchHit b) {
    final aEmbedding = a.chunk.embedding;
    final bEmbedding = b.chunk.embedding;
    if (aEmbedding != null &&
        bEmbedding != null &&
        aEmbedding.length == bEmbedding.length) {
      return ((_cosineSimilarity(aEmbedding, bEmbedding) + 1) / 2).clamp(
        0.0,
        1.0,
      );
    }

    final aTerms = _tokenize(a.chunk.content).toSet();
    final bTerms = _tokenize(b.chunk.content).toSet();
    if (aTerms.isEmpty || bTerms.isEmpty) return 0;

    final intersection = aTerms.intersection(bTerms).length;
    final union = aTerms.union(bTerms).length;
    return union == 0 ? 0 : intersection / union;
  }

  double _cosineSimilarity(
    List<double> a,
    List<double> b, {
    double? normA,
    double? normB,
  }) {
    final aNorm = normA ?? _vectorNorm(a);
    final bNorm = normB ?? _vectorNorm(b);
    if (aNorm == 0 || bNorm == 0) return 0;

    var dot = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }

    return dot / (aNorm * bNorm);
  }

  double _vectorNorm(List<double> values) {
    var norm = 0.0;
    for (final value in values) {
      norm += value * value;
    }
    return norm == 0 ? 0 : math.sqrt(norm);
  }

  double _maxScore(Iterable<double> scores) {
    var maxScore = 0.0;
    for (final score in scores) {
      if (score.isFinite && score > maxScore) maxScore = score;
    }
    return maxScore;
  }

  double _normalizeScore(double score, double maxScore) {
    if (!score.isFinite || score <= 0 || maxScore <= 0) return 0;
    return (score / maxScore).clamp(0.0, 1.0);
  }

  double _coverageScore(List<String> queryTerms, Set<String> evidenceTerms) {
    if (queryTerms.isEmpty || evidenceTerms.isEmpty) return 0;
    final uniqueQueryTerms = queryTerms.toSet();
    final matches = uniqueQueryTerms.where(evidenceTerms.contains).length;
    return matches / uniqueQueryTerms.length;
  }

  double _phraseScore(List<String> phrases, String normalizedEvidence) {
    if (phrases.isEmpty || normalizedEvidence.isEmpty) return 0;
    final matches = phrases
        .where((phrase) => normalizedEvidence.contains(phrase))
        .length;
    return matches / phrases.length;
  }

  double _exactSetScore(Set<String> queryValues, Set<String> evidenceValues) {
    if (queryValues.isEmpty || evidenceValues.isEmpty) return 0;
    final matches = queryValues.intersection(evidenceValues).length;
    return matches / queryValues.length;
  }

  double _proximityScore(List<String> queryTerms, List<String> evidenceTerms) {
    if (queryTerms.isEmpty || evidenceTerms.isEmpty) return 0;
    final requested = queryTerms.toSet();
    final evidenceTermSet = evidenceTerms.toSet();
    final matched = requested.where(evidenceTermSet.contains).toSet();
    if (matched.isEmpty) return 0;
    if (matched.length == 1) return 0.25;

    final counts = <String, int>{};
    var covered = 0;
    var left = 0;
    var bestSpan = evidenceTerms.length + 1;

    for (var right = 0; right < evidenceTerms.length; right++) {
      final rightTerm = evidenceTerms[right];
      if (matched.contains(rightTerm)) {
        final count = counts[rightTerm] ?? 0;
        counts[rightTerm] = count + 1;
        if (count == 0) covered++;
      }

      while (covered == matched.length && left <= right) {
        bestSpan = math.min(bestSpan, right - left + 1);
        final leftTerm = evidenceTerms[left];
        if (matched.contains(leftTerm)) {
          final count = counts[leftTerm] ?? 0;
          if (count <= 1) {
            counts.remove(leftTerm);
            covered--;
          } else {
            counts[leftTerm] = count - 1;
          }
        }
        left++;
      }
    }

    if (bestSpan > evidenceTerms.length) return 0;
    final coverage = matched.length / requested.length;
    final tightness = matched.length / bestSpan;
    return (coverage * math.sqrt(tightness)).clamp(0.0, 1.0);
  }

  double _fieldCueScore(Set<String> fieldCues, String normalizedEvidence) {
    if (fieldCues.isEmpty || normalizedEvidence.isEmpty) return 0;
    final matches = fieldCues
        .where((cue) => normalizedEvidence.contains(cue))
        .length;
    return matches / fieldCues.length;
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
          'preview': AppLogger.preview(hits[i].chunk.content, 140),
        },
    ];
  }
}

class _TokenizedCandidate {
  _TokenizedCandidate({
    required this.candidate,
    required this.terms,
    required this.titleTerms,
  }) : frequencies = terms.groupFoldBy<String, int>(
         (term) => term,
         (count, _) => (count ?? 0) + 1,
       ),
       uniqueTerms = terms.toSet();

  final RetrievalCandidate candidate;
  final List<String> terms;
  final Set<String> titleTerms;
  final Map<String, int> frequencies;
  final Set<String> uniqueTerms;

  int get length => terms.length;
}

class _FusionEntry {
  _FusionEntry({required this.document, required this.chunk});

  final IndexedDocument document;
  final DocumentChunk chunk;
  double score = 0;
  double keywordScore = 0;
  double vectorScore = 0;
}

class _RerankedHit {
  const _RerankedHit({
    required this.hit,
    required this.firstStageScore,
    required this.coverageScore,
    required this.phraseScore,
    required this.identifierScore,
    required this.numberScore,
    required this.titleScore,
    required this.proximityScore,
    required this.fieldCueScore,
  });

  final SearchHit hit;
  final double firstStageScore;
  final double coverageScore;
  final double phraseScore;
  final double identifierScore;
  final double numberScore;
  final double titleScore;
  final double proximityScore;
  final double fieldCueScore;

  Map<String, Object?> summary(int index) {
    return {
      'rank': index + 1,
      'document': hit.document.name,
      'chunkIndex': hit.chunk.index,
      'rerankScore': AppLogger.score(hit.score),
      'firstStageScore': AppLogger.score(firstStageScore),
      'coverageScore': AppLogger.score(coverageScore),
      'phraseScore': AppLogger.score(phraseScore),
      'identifierScore': AppLogger.score(identifierScore),
      'numberScore': AppLogger.score(numberScore),
      'titleScore': AppLogger.score(titleScore),
      'proximityScore': AppLogger.score(proximityScore),
      'fieldCueScore': AppLogger.score(fieldCueScore),
      'keywordScore': AppLogger.score(hit.keywordScore),
      'vectorScore': AppLogger.score(hit.vectorScore),
      'preview': AppLogger.preview(hit.chunk.content, 140),
    };
  }
}

class _QueryIntent {
  const _QueryIntent({
    required this.queryTerms,
    required this.phrases,
    required this.identifiers,
    required this.numbers,
    required this.fieldCues,
  });

  factory _QueryIntent.fromQuery(String query) {
    final queryTerms = _dedupe(_tokenize(query));
    return _QueryIntent(
      queryTerms: queryTerms,
      phrases: _queryPhrases(queryTerms),
      identifiers: _extractIdentifiers(query),
      numbers: _extractNumbers(query),
      fieldCues: _fieldCuesForQuery(query),
    );
  }

  final List<String> queryTerms;
  final List<String> phrases;
  final Set<String> identifiers;
  final Set<String> numbers;
  final Set<String> fieldCues;
}

enum _RetrievalSignal { keyword, vector }

List<String> _tokenize(String value) {
  return RegExp(r'[a-zA-Z0-9_]{2,}')
      .allMatches(value.toLowerCase())
      .map((match) => match.group(0)!)
      .where((term) => !_stopWords.contains(term))
      .toList(growable: false);
}

List<String> _dedupe(Iterable<String> values) {
  final seen = <String>{};
  final result = <String>[];
  for (final value in values) {
    if (seen.add(value)) result.add(value);
  }
  return result;
}

List<String> _queryPhrases(List<String> queryTerms) {
  if (queryTerms.length < 2) return const [];
  final phrases = <String>[];
  for (final width in [3, 2]) {
    if (queryTerms.length < width) continue;
    for (var i = 0; i <= queryTerms.length - width; i++) {
      final window = queryTerms.skip(i).take(width).toList(growable: false);
      if (window.any(_isMostlyNumeric)) continue;
      phrases.add(window.join(' '));
    }
  }
  return _dedupe(phrases);
}

Set<String> _extractIdentifiers(String value) {
  final pattern = RegExp(
    r'\b(?:[a-z]{2,}[a-z0-9]*[-_][a-z0-9][a-z0-9_-]*|\d{4}[-/]\d{1,2}[-/]\d{1,2}|[a-z]{2,}\d{2,})\b',
    caseSensitive: false,
  );
  return pattern
      .allMatches(value)
      .map((match) => _normalizeIdentifier(match.group(0)!))
      .where((identifier) => identifier.length >= 4)
      .toSet();
}

Set<String> _extractNumbers(String value) {
  final pattern = RegExp(
    r'(?:^|[^a-z0-9])([$₹€£]?\s*\d[\d,]*(?:\.\d+)?%?)',
    caseSensitive: false,
  );
  return pattern
      .allMatches(value.toLowerCase())
      .map((match) => _normalizeNumber(match.group(1)!))
      .where((number) => number.isNotEmpty)
      .toSet();
}

String _normalizeIdentifier(String value) {
  return value.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');
}

String _normalizeNumber(String value) {
  final normalized = value.replaceAll(RegExp(r'[^0-9.]'), '');
  return normalized.replaceFirst(RegExp(r'\.0+$'), '');
}

String _normalizeForPhrase(String value) {
  return value
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z0-9]+'), ' ')
      .replaceAll(RegExp(r'\s+'), ' ')
      .trim();
}

bool _isMostlyNumeric(String value) {
  final digitCount = RegExp(r'\d').allMatches(value).length;
  return digitCount > 0 && digitCount >= value.length / 2;
}

Set<String> _fieldCuesForQuery(String query) {
  final normalized = _normalizeForPhrase(query);
  final cues = <String>{};

  void addIf(bool condition, Iterable<String> values) {
    if (condition) cues.addAll(values);
  }

  addIf(normalized.contains('owner') || normalized.contains('team'), const [
    'owner',
    'owner team',
  ]);
  addIf(
    normalized.contains('approval') || normalized.contains('approved'),
    const ['approval', 'approval path'],
  );
  addIf(
    normalized.contains('deadline') ||
        normalized.contains('due') ||
        normalized.contains('date'),
    const ['deadline', 'date'],
  );
  addIf(
    normalized.contains('budget') ||
        normalized.contains('amount') ||
        normalized.contains('cost'),
    const ['budget', 'amount', 'cost'],
  );
  addIf(
    normalized.contains('where') ||
        normalized.contains('location') ||
        normalized.contains('held') ||
        normalized.contains('district') ||
        normalized.contains('region'),
    const ['location', 'venue', 'held', 'district', 'region', 'site'],
  );
  addIf(
    normalized.contains('status') ||
        normalized.contains('draft') ||
        normalized.contains('planned'),
    const ['status', 'draft', 'planned', 'upgrade'],
  );
  addIf(
    normalized.contains('recorded') ||
        normalized.contains('count') ||
        normalized.contains('trips'),
    const ['recorded', 'counter', 'count', 'trips'],
  );
  addIf(normalized.contains('rate') || normalized.contains('percent'), const [
    'rate',
    'percent',
    'percentage',
  ]);
  addIf(normalized.contains('severity'), const ['severity', 'severity class']);
  addIf(normalized.contains('retention'), const [
    'retention',
    'retention window',
  ]);

  return cues;
}

const Set<String> _stopWords = {
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'how',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'the',
  'this',
  'to',
  'was',
  'what',
  'when',
  'where',
  'which',
  'who',
  'why',
  'with',
};
