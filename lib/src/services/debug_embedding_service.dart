import 'dart:math' as math;

class DebugEmbeddingService {
  const DebugEmbeddingService({this.dimension = 384});

  final int dimension;

  List<List<double>> embedDocuments(
    List<String> texts, {
    required int maxInputChars,
  }) {
    return texts
        .map((text) => embedText(text, maxInputChars: maxInputChars))
        .toList(growable: false);
  }

  List<double> embedQuery(String text, {required int maxInputChars}) {
    return embedText(text, maxInputChars: maxInputChars);
  }

  List<double> embedText(String text, {required int maxInputChars}) {
    final vector = List<double>.filled(dimension, 0);
    final clipped = _clip(text, maxInputChars);
    final terms = _tokenize(clipped);
    if (terms.isEmpty) return vector;

    for (final term in terms) {
      _addFeature(vector, 't:$term', 1.0);
      final stem = _stem(term);
      if (stem != term) {
        _addFeature(vector, 's:$stem', 0.85);
      }
    }

    for (var i = 0; i < terms.length - 1; i++) {
      _addFeature(vector, 'b:${terms[i]} ${terms[i + 1]}', 0.65);
    }

    return _normalize(vector);
  }

  String _clip(String text, int maxInputChars) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (maxInputChars < 1 || normalized.length <= maxInputChars) {
      return normalized;
    }
    return normalized.substring(0, maxInputChars).trimRight();
  }

  List<String> _tokenize(String value) {
    return RegExp(r'[a-zA-Z0-9_]{2,}')
        .allMatches(value.toLowerCase())
        .map((match) => match.group(0)!)
        .where((term) => !_stopWords.contains(term))
        .toList(growable: false);
  }

  String _stem(String term) {
    if (term.length > 5 && term.endsWith('ing')) {
      return term.substring(0, term.length - 3);
    }
    if (term.length > 4 && term.endsWith('ies')) {
      return '${term.substring(0, term.length - 3)}y';
    }
    if (term.length > 4 && term.endsWith('es')) {
      return term.substring(0, term.length - 2);
    }
    if (term.length > 3 && term.endsWith('s')) {
      return term.substring(0, term.length - 1);
    }
    if (term.length > 4 && term.endsWith('ed')) {
      return term.substring(0, term.length - 2);
    }
    return term;
  }

  void _addFeature(List<double> vector, String feature, double weight) {
    final hash = _fnv1a32(feature);
    final index = hash % dimension;
    final sign = (hash & 0x80000000) == 0 ? 1.0 : -1.0;
    vector[index] += sign * weight;
  }

  int _fnv1a32(String value) {
    var hash = 0x811c9dc5;
    for (final unit in value.codeUnits) {
      hash ^= unit;
      hash = (hash * 0x01000193) & 0xffffffff;
    }
    return hash;
  }

  List<double> _normalize(List<double> vector) {
    var norm = 0.0;
    for (final value in vector) {
      norm += value * value;
    }
    if (norm == 0) return vector;

    final scale = 1 / math.sqrt(norm);
    for (var i = 0; i < vector.length; i++) {
      vector[i] *= scale;
    }
    return vector;
  }
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
