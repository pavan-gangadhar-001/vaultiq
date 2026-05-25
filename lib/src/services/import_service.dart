import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:saf/saf.dart';

class ImportCandidate {
  const ImportCandidate({
    required this.name,
    required this.sourcePath,
    this.path,
    this.bytes,
    this.readStream,
  });

  final String name;
  final String sourcePath;
  final String? path;
  final List<int>? bytes;
  final Stream<List<int>>? readStream;
}

class ImportService {
  static const supportedPickerExtensions = [
    'txt',
    'md',
    'markdown',
    'csv',
    'json',
    'log',
    'yaml',
    'yml',
    'xml',
    'html',
    'htm',
    'css',
    'js',
    'ts',
    'dart',
    'py',
    'java',
    'kt',
    'swift',
    'go',
    'rs',
    'c',
    'cpp',
    'h',
    'hpp',
    'sql',
    'pdf',
    'docx',
  ];

  Future<List<ImportCandidate>> pickFiles() async {
    final result = await FilePicker.pickFiles(
      allowMultiple: true,
      withData: false,
      withReadStream: true,
      type: Platform.isAndroid ? FileType.any : FileType.custom,
      allowedExtensions: Platform.isAndroid ? null : supportedPickerExtensions,
    );
    if (result == null) return const [];

    return result.files
        .map(
          (file) => ImportCandidate(
            name: file.name,
            sourcePath: file.path ?? file.identifier ?? file.name,
            path: file.path,
            bytes: file.bytes,
            readStream: file.readStream,
          ),
        )
        .toList(growable: false);
  }

  Future<List<ImportCandidate>> pickFolder() async {
    if (Platform.isAndroid) {
      return _pickAndroidFolder();
    }

    final path = await FilePicker.getDirectoryPath();
    if (path == null) return const [];

    final directory = Directory(path);
    if (!directory.existsSync()) return const [];

    return directory
        .listSync(recursive: true, followLinks: false)
        .whereType<File>()
        .map(
          (file) => ImportCandidate(
            name: file.uri.pathSegments.last,
            sourcePath: file.path,
            path: file.path,
          ),
        )
        .toList(growable: false);
  }

  Future<List<ImportCandidate>> _pickAndroidFolder() async {
    final before = await Saf.getPersistedPermissionDirectories() ?? const [];
    final granted = await Saf.getDynamicDirectoryPermission(
      grantWritePermission: false,
    );
    if (granted != true) return const [];

    final after = await Saf.getPersistedPermissionDirectories() ?? const [];
    final selected = after.firstWhere(
      (directory) => !before.contains(directory),
      orElse: () => after.isNotEmpty ? after.last : '',
    );
    if (selected.isEmpty) return const [];

    final cached = await Saf.cacheFor(selected, fileType: FileTypes.any);
    if (cached == null || cached.isEmpty) return const [];

    return cached
        .map(
          (path) => ImportCandidate(
            name: path.split('/').last,
            sourcePath: path,
            path: path,
          ),
        )
        .toList(growable: false);
  }
}
