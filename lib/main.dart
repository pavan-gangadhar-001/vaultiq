import 'package:flutter/material.dart';

import 'src/app_controller.dart';
import 'src/screens/home_screen.dart';
import 'src/services/document_store.dart';
import 'src/services/import_service.dart';
import 'src/services/local_ai_service.dart';
import 'src/services/text_extractor.dart';

void main() {
  runApp(const LocalDocQaApp());
}

class LocalDocQaApp extends StatefulWidget {
  const LocalDocQaApp({super.key});

  @override
  State<LocalDocQaApp> createState() => _LocalDocQaAppState();
}

class _LocalDocQaAppState extends State<LocalDocQaApp> {
  late final AppController controller;

  @override
  void initState() {
    super.initState();
    controller = AppController(
      store: DocumentStore(),
      importService: ImportService(),
      extractor: TextExtractor(),
      aiService: LocalAiService(),
    )..initialize();
  }

  @override
  void dispose() {
    controller.disposeApp();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'VaultIQ',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme:
            ColorScheme.fromSeed(
              seedColor: const Color(0xFF00BFA6),
              brightness: Brightness.light,
            ).copyWith(
              primary: const Color(0xFF008B7A),
              secondary: const Color(0xFFFFB454),
              tertiary: const Color(0xFF2A7FFF),
              surface: Colors.white,
            ),
        scaffoldBackgroundColor: const Color(0xFF08161C),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          foregroundColor: Colors.white,
          elevation: 0,
          surfaceTintColor: Colors.transparent,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: const BorderSide(color: Color(0xFFE6F0ED)),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFF00A993),
            foregroundColor: Colors.white,
            minimumSize: const Size(48, 48),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFF06343A),
            minimumSize: const Size(48, 48),
            side: const BorderSide(color: Color(0xFF99DED3)),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          filled: true,
          fillColor: const Color(0xFFF8FCFA),
        ),
      ),
      home: HomeScreen(controller: controller),
    );
  }
}
