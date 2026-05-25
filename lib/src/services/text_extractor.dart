import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:path/path.dart' as p;
import 'package:syncfusion_flutter_pdf/pdf.dart';
import 'package:xml/xml.dart';

import '../models.dart';
import 'app_logger.dart';
import 'import_service.dart';

class TextExtractor {
  static const int maxFileBytes = 25 * 1024 * 1024;

  static const Set<String> supportedExtensions = {
    '.txt',
    '.md',
    '.markdown',
    '.csv',
    '.json',
    '.log',
    '.yaml',
    '.yml',
    '.xml',
    '.html',
    '.htm',
    '.css',
    '.js',
    '.ts',
    '.dart',
    '.py',
    '.java',
    '.kt',
    '.swift',
    '.go',
    '.rs',
    '.c',
    '.cpp',
    '.h',
    '.hpp',
    '.sql',
    '.pdf',
    '.docx',
  };

  bool isSupported(String fileName) {
    return supportedExtensions.contains(p.extension(fileName).toLowerCase());
  }

  Future<ExtractedDocument> extract(ImportCandidate candidate) async {
    final stopwatch = Stopwatch()..start();
    final extension = p.extension(candidate.name).toLowerCase();
    AppLogger.info('extract.start', {
      'name': candidate.name,
      'sourcePath': candidate.sourcePath,
      'extension': extension,
      'hasInlineBytes': candidate.bytes != null,
      'hasPath': candidate.path != null,
      'hasReadStream': candidate.readStream != null,
    });

    try {
      final bytes = candidate.bytes ?? await _readFile(candidate);
      if (bytes.length > maxFileBytes) {
        throw const FormatException(
          'file is larger than the 25 MB import limit',
        );
      }

      final text = switch (extension) {
        '.pdf' => _extractPdf(bytes),
        '.docx' => _extractDocx(bytes),
        _ => _normalizeText(utf8.decode(bytes, allowMalformed: true)),
      };

      AppLogger.info('extract.done', {
        'name': candidate.name,
        'extension': extension,
        'bytes': bytes.length,
        'textChars': text.length,
        'textPreview': AppLogger.preview(text),
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      return ExtractedDocument(
        name: candidate.name,
        sourcePath: candidate.sourcePath,
        bytes: bytes.length,
        text: text,
      );
    } catch (e, st) {
      AppLogger.error('extract.error', e, st, {
        'name': candidate.name,
        'extension': extension,
        'durationMs': stopwatch.elapsedMilliseconds,
      });
      rethrow;
    }
  }

  Future<List<int>> _readFile(ImportCandidate candidate) async {
    final path = candidate.path;
    if (path != null) {
      final file = File(path);
      final length = await file.length();
      if (length > maxFileBytes) {
        throw const FormatException(
          'file is larger than the 25 MB import limit',
        );
      }
      return file.readAsBytes();
    }

    final readStream = candidate.readStream;
    if (readStream == null) {
      throw const FileSystemException('file path is unavailable');
    }

    final builder = BytesBuilder(copy: false);
    var length = 0;
    await for (final chunk in readStream) {
      length += chunk.length;
      AppLogger.info('extract.stream_chunk.read', {
        'name': candidate.name,
        'chunkBytes': chunk.length,
        'totalBytes': length,
      });
      if (length > maxFileBytes) {
        throw const FormatException(
          'file is larger than the 25 MB import limit',
        );
      }
      builder.add(chunk);
    }
    return builder.takeBytes();
  }

  String _extractPdf(List<int> bytes) {
    final document = PdfDocument(inputBytes: bytes);
    try {
      return _normalizeText(PdfTextExtractor(document).extractText());
    } finally {
      document.dispose();
    }
  }

  String _extractDocx(List<int> bytes) {
    final archive = ZipDecoder().decodeBytes(bytes, verify: true);
    final documentFile = archive.findFile('word/document.xml');
    if (documentFile == null) {
      throw const FormatException('DOCX document.xml is missing');
    }

    final documentXml = utf8.decode(documentFile.content, allowMalformed: true);
    final document = XmlDocument.parse(documentXml);
    final buffer = StringBuffer();

    for (final paragraph in document.descendants.whereType<XmlElement>()) {
      if (paragraph.name.local != 'p') continue;

      final paragraphText = StringBuffer();
      for (final element in paragraph.descendants.whereType<XmlElement>()) {
        switch (element.name.local) {
          case 't':
            paragraphText.write(element.innerText);
            break;
          case 'tab':
            paragraphText.write('\t');
            break;
          case 'br':
          case 'cr':
            paragraphText.write('\n');
            break;
        }
      }

      final normalizedParagraph = paragraphText.toString().trim();
      if (normalizedParagraph.isNotEmpty) {
        buffer.writeln(normalizedParagraph);
      }
    }

    final text = _normalizeText(buffer.toString());
    if (text.isEmpty) {
      throw const FormatException('DOCX did not contain readable text');
    }
    return text;
  }

  String _normalizeText(String value) {
    return value
        .replaceAll('\u0000', ' ')
        .replaceAll(RegExp(r'[ \t]+'), ' ')
        .replaceAll(RegExp(r'\n{3,}'), '\n\n')
        .trim();
  }
}
