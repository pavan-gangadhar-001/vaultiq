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
    this.candidateMultiplier = 5,
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

    final queryTerms = _tokenize(query).toSet();
    if (queryTerms.isEmpty && queryEmbedding == null) {
      AppLogger.warn('ranker.rank.no_signals', {
        'queryPreview': AppLogger.preview(query),
      });
      return const [];
    }

    final poolSize = math.max(topK, topK * candidateMultiplier);
    final keywordHits = queryTerms.isEmpty
        ? <SearchHit>[]
        : _rankBm25(
            queryTerms: queryTerms,
            candidates: candidates,
          ).take(poolSize).toList(growable: false);
    final vectorHits = queryEmbedding == null
        ? <SearchHit>[]
        : _rankVectors(
            queryEmbedding: queryEmbedding,
            candidates: candidates,
          ).take(poolSize).toList(growable: false);

    final ranked = vectorHits.isEmpty
        ? keywordHits
        : _fuseRankings(keywordHits: keywordHits, vectorHits: vectorHits);

    final diversified = _diversify(ranked, topK).toList(growable: false);
    AppLogger.info('ranker.rank.done', {
      'durationMs': stopwatch.elapsedMilliseconds,
      'queryTerms': queryTerms.toList(),
      'poolSize': poolSize,
      'keywordHitCount': keywordHits.length,
      'vectorHitCount': vectorHits.length,
      'rankedCount': ranked.length,
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

enum _RetrievalSignal { keyword, vector }

List<String> _tokenize(String value) {
  return RegExp(r'[a-zA-Z0-9_]{2,}')
      .allMatches(value.toLowerCase())
      .map((match) => match.group(0)!)
      .where((term) => !_stopWords.contains(term))
      .toList(growable: false);
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
