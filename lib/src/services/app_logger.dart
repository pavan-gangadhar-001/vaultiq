import 'dart:convert';

import 'package:flutter/foundation.dart';

class AppLogger {
  AppLogger._();

  static const String tag = 'LocalDocQA';
  static int _sequence = 0;

  static void info(String event, [Map<String, Object?> data = const {}]) {
    _write('INFO', event, data);
  }

  static void warn(String event, [Map<String, Object?> data = const {}]) {
    _write('WARN', event, data);
  }

  static void error(
    String event,
    Object error, [
    StackTrace? stackTrace,
    Map<String, Object?> data = const {},
  ]) {
    _write('ERROR', event, {
      ...data,
      'error': error.toString(),
      if (stackTrace != null) 'stack': preview(stackTrace.toString(), 900),
    });
  }

  static String preview(String text, [int maxChars = 180]) {
    final normalized = _redact(text.replaceAll(RegExp(r'\s+'), ' ').trim());
    if (normalized.length <= maxChars) return normalized;
    if (maxChars <= 3) return normalized.substring(0, maxChars);
    return '${normalized.substring(0, maxChars - 3).trimRight()}...';
  }

  static Object score(double value) {
    if (!value.isFinite) return value.toString();
    return double.parse(value.toStringAsFixed(4));
  }

  static Map<String, Object?> textStats(Iterable<String> values) {
    final lengths = values.map((value) => value.length).toList(growable: false);
    if (lengths.isEmpty) {
      return const {'count': 0, 'minChars': 0, 'maxChars': 0, 'avgChars': 0};
    }
    final total = lengths.fold<int>(0, (sum, length) => sum + length);
    lengths.sort();
    return {
      'count': lengths.length,
      'minChars': lengths.first,
      'maxChars': lengths.last,
      'avgChars': double.parse((total / lengths.length).toStringAsFixed(1)),
    };
  }

  static void _write(String level, String event, Map<String, Object?> data) {
    final payload = <String, Object?>{
      'seq': ++_sequence,
      'time': DateTime.now().toIso8601String(),
      'level': level,
      'event': event,
      ..._sanitizeMap(data),
    };

    debugPrint('[$tag] ${jsonEncode(payload)}', wrapWidth: 1200);
  }

  static Map<String, Object?> _sanitizeMap(Map<String, Object?> value) {
    return {
      for (final entry in value.entries) entry.key: _sanitize(entry.value),
    };
  }

  static Object? _sanitize(Object? value) {
    if (value == null || value is num || value is bool) {
      return value;
    }
    if (value is String) return _redact(value);
    if (value is DateTime) return value.toIso8601String();
    if (value is Duration) return value.inMilliseconds;
    if (value is Map) {
      return {
        for (final entry in value.entries)
          entry.key.toString(): _sanitize(entry.value),
      };
    }
    if (value is Iterable) {
      return value.take(25).map(_sanitize).toList(growable: false);
    }
    return _redact(value.toString());
  }

  static String _redact(String value) {
    var sanitized = value;
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
}
