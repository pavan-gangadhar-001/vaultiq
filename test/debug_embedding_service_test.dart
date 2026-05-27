import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:vaultiq/src/model_catalog.dart';
import 'package:vaultiq/src/services/debug_embedding_service.dart';
import 'package:vaultiq/src/services/local_ai_service.dart';

void main() {
  test('debug embedder produces deterministic normalized vectors', () {
    const service = DebugEmbeddingService(dimension: 64);

    final first = service.embedQuery(
      'Lunch credits are meal benefits.',
      maxInputChars: 200,
    );
    final second = service.embedQuery(
      'Lunch credits are meal benefits.',
      maxInputChars: 200,
    );

    expect(first, second);
    expect(first, hasLength(64));
    expect(_norm(first), closeTo(1, 0.000001));
  });

  test('debug embedder gives related text a higher cosine score', () {
    const service = DebugEmbeddingService(dimension: 384);

    final query = service.embedQuery(
      'Where are meal benefits and lunch credits described?',
      maxInputChars: 400,
    );
    final related = service.embedText(
      'Cafeteria lunch credits are a workplace meal benefit.',
      maxInputChars: 400,
    );
    final unrelated = service.embedText(
      'Android Gradle builds require matching SDK platforms.',
      maxInputChars: 400,
    );

    expect(_cosine(query, related), greaterThan(_cosine(query, unrelated)));
  });

  test('embedder selection uses built-in fallback for x86 emulators', () {
    final service = LocalAiService();

    expect(
      service.embeddingModelForAbis(const ['x86_64', 'arm64-v8a']),
      DownloadableEmbeddingModel.debugHashing,
    );
    expect(
      service.embeddingModelForAbis(const ['arm64-v8a']),
      DownloadableEmbeddingModel.gecko256,
    );
  });
}

double _norm(List<double> values) {
  var sum = 0.0;
  for (final value in values) {
    sum += value * value;
  }
  return math.sqrt(sum);
}

double _cosine(List<double> a, List<double> b) {
  var dot = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot / (_norm(a) * _norm(b));
}
