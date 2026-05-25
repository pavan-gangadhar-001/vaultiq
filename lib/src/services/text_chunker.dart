import 'app_logger.dart';

class TextChunker {
  const TextChunker({this.chunkSize = 900, this.overlap = 120});

  final int chunkSize;
  final int overlap;

  List<String> chunk(String text) {
    final stopwatch = Stopwatch()..start();
    final normalized = text.trim();
    AppLogger.info('chunker.chunk.start', {
      'inputChars': text.length,
      'normalizedChars': normalized.length,
      'chunkSize': chunkSize,
      'overlap': overlap,
    });
    if (normalized.isEmpty) {
      AppLogger.warn('chunker.chunk.empty', {
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      return const [];
    }
    if (normalized.length <= chunkSize) {
      AppLogger.info('chunker.chunk.done', {
        'chunkCount': 1,
        'chunkStats': AppLogger.textStats([normalized]),
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      return [normalized];
    }

    final chunks = <String>[];
    var start = 0;
    while (start < normalized.length) {
      var end = (start + chunkSize).clamp(0, normalized.length);
      if (end < normalized.length) {
        final paragraphBreak = normalized.lastIndexOf('\n\n', end);
        final sentenceBreak = normalized.lastIndexOf(RegExp(r'[.!?]\s'), end);
        final candidateBreak =
            [paragraphBreak, sentenceBreak == -1 ? -1 : sentenceBreak + 1]
                .where((index) => index > start + chunkSize ~/ 2)
                .fold<int>(-1, (best, index) => index > best ? index : best);
        if (candidateBreak != -1) {
          end = candidateBreak;
        }
      }

      final chunk = normalized.substring(start, end).trim();
      if (chunk.isNotEmpty) chunks.add(chunk);

      if (end >= normalized.length) break;
      start = (end - overlap).clamp(0, normalized.length);
    }

    AppLogger.info('chunker.chunk.done', {
      'chunkCount': chunks.length,
      'chunkStats': AppLogger.textStats(chunks),
      'sampleChunks': [
        for (var i = 0; i < chunks.length && i < 3; i++)
          {
            'index': i,
            'chars': chunks[i].length,
            'preview': AppLogger.preview(chunks[i], 140),
          },
      ],
      'durationMs': stopwatch.elapsedMilliseconds,
    });
    return chunks;
  }
}
