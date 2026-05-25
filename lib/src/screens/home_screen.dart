import 'package:flutter/material.dart';

import '../app_controller.dart';
import '../models.dart';

const _appName = 'VaultIQ';
const _brandIconAsset = 'assets/brand/vaultiq_icon.png';

class _BrandColors {
  static const ink = Color(0xFF08161C);
  static const deepTeal = Color(0xFF06343A);
  static const teal = Color(0xFF00A993);
  static const mint = Color(0xFF62F0CF);
  static const amber = Color(0xFFFFB454);
  static const coral = Color(0xFFFF6B5F);
  static const sky = Color(0xFF7CC7FF);
  static const panel = Color(0xF7FFFFFF);
  static const panelSoft = Color(0xFFEFFAF6);
  static const border = Color(0xFFE0F1EC);
}

const _screenDecoration = BoxDecoration(
  gradient: LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      _BrandColors.ink,
      _BrandColors.deepTeal,
      Color(0xFF0B665E),
      Color(0xFF172129),
    ],
    stops: [0, 0.38, 0.7, 1],
  ),
);

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.controller});

  final AppController controller;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _questionController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  AppController get controller => widget.controller;

  @override
  void dispose() {
    _questionController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        return Scaffold(
          appBar: AppBar(
            titleSpacing: 16,
            title: const _BrandTitle(),
            actions: [
              IconButton(
                tooltip: 'Local AI setup',
                onPressed: _showSetupSheet,
                icon: Icon(
                  controller.dependenciesReady
                      ? Icons.offline_bolt
                      : Icons.offline_bolt_outlined,
                ),
              ),
              IconButton(
                tooltip: 'Clear index',
                onPressed: controller.stats.documentCount == 0
                    ? null
                    : _confirmClear,
                icon: const Icon(Icons.delete_outline),
              ),
            ],
          ),
          body: DecoratedBox(
            decoration: _screenDecoration,
            child: SafeArea(
              child: controller.isInitializing
                  ? const Center(
                      child: CircularProgressIndicator(color: Colors.white),
                    )
                  : controller.needsSetup
                  ? _SetupPage(controller: controller)
                  : Column(
                      children: [
                        _StatusHeader(controller: controller),
                        _ImportToolbar(controller: controller),
                        Expanded(
                          child: CustomScrollView(
                            controller: _scrollController,
                            slivers: [
                              SliverToBoxAdapter(
                                child: _DocumentStrip(controller: controller),
                              ),
                              SliverPadding(
                                padding: const EdgeInsets.fromLTRB(
                                  16,
                                  8,
                                  16,
                                  8,
                                ),
                                sliver: SliverList.separated(
                                  itemBuilder: (context, index) => _ChatBubble(
                                    turn: controller.messages[index],
                                  ),
                                  separatorBuilder: (_, _) =>
                                      const SizedBox(height: 10),
                                  itemCount: controller.messages.length,
                                ),
                              ),
                            ],
                          ),
                        ),
                        _QuestionBar(
                          controller: _questionController,
                          enabled:
                              !controller.isAnswering &&
                              !controller.isInstallingDependencies &&
                              !controller.isIndexing &&
                              controller.dependenciesReady &&
                              controller.stats.documentCount > 0,
                          hintText: _questionHint(controller),
                          onSubmit: _submitQuestion,
                        ),
                      ],
                    ),
            ),
          ),
        );
      },
    );
  }

  String _questionHint(AppController controller) {
    if (controller.isInstallingDependencies) return 'Finishing local AI setup';
    if (controller.isIndexing) return 'Indexing local files';
    if (!controller.dependenciesReady) return 'Install local AI before asking';
    if (controller.stats.documentCount == 0) {
      return 'Import files before asking';
    }
    return 'Ask about your local files';
  }

  Future<void> _submitQuestion() async {
    final question = _questionController.text;
    _questionController.clear();
    await controller.ask(question);
    if (!mounted) return;
    await Future<void>.delayed(const Duration(milliseconds: 50));
    if (_scrollController.hasClients) {
      await _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 260),
        curve: Curves.easeOutCubic,
      );
    }
  }

  Future<void> _showSetupSheet() {
    return showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (context) => AnimatedBuilder(
        animation: controller,
        builder: (context, _) => SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
            child: _DependencySetupContent(controller: controller),
          ),
        ),
      ),
    );
  }

  Future<void> _confirmClear() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear local index?'),
        content: const Text(
          'Imported source files stay where they are. Only the app index is removed.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await controller.clearIndex();
    }
  }
}

class _BrandTitle extends StatelessWidget {
  const _BrandTitle();

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Image.asset(
            _brandIconAsset,
            width: 34,
            height: 34,
            fit: BoxFit.cover,
          ),
        ),
        const SizedBox(width: 10),
        const Text(_appName, style: TextStyle(fontWeight: FontWeight.w800)),
      ],
    );
  }
}

class _SetupPage extends StatelessWidget {
  const _SetupPage({required this.controller});

  final AppController controller;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: _DependencySetupContent(controller: controller),
        ),
      ),
    );
  }
}

class _DependencySetupContent extends StatelessWidget {
  const _DependencySetupContent({required this.controller});

  final AppController controller;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final answerProgress = controller.modelDownloadProgress;
    final semanticProgress =
        controller.embeddingModelDownloadProgress == null &&
            controller.embeddingTokenizerDownloadProgress == null
        ? null
        : ((controller.embeddingModelDownloadProgress ?? 0) +
                  (controller.embeddingTokenizerDownloadProgress ?? 0)) /
              200;
    final busy = controller.isInstallingDependencies;
    final ready = controller.dependenciesReady;

    return DecoratedBox(
      decoration: BoxDecoration(
        color: _BrandColors.panel,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0x66FFFFFF)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x33000000),
            blurRadius: 28,
            offset: Offset(0, 18),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(22),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Center(
              child: Container(
                width: 92,
                height: 92,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [_BrandColors.mint, _BrandColors.amber],
                  ),
                  borderRadius: BorderRadius.circular(8),
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x5500BFA6),
                      blurRadius: 24,
                      offset: Offset(0, 12),
                    ),
                  ],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.asset(_brandIconAsset, fit: BoxFit.cover),
                ),
              ),
            ),
            const SizedBox(height: 18),
            Text(
              ready ? 'VaultIQ is ready' : 'Set up VaultIQ',
              textAlign: TextAlign.center,
              style: theme.textTheme.headlineSmall?.copyWith(
                color: _BrandColors.deepTeal,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              ready
                  ? 'Everything needed for private document QA is installed on this device.'
                  : 'Install the local answer engine and semantic search files before using the app.',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: const Color(0xFF315158),
              ),
            ),
            if (!controller.supportsEmbeddingModel) ...[
              const SizedBox(height: 12),
              Text(
                'Semantic search is not available on this device. Keyword search will still work after setup.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ],
            const SizedBox(height: 24),
            _DependencyRow(
              icon: Icons.psychology_outlined,
              title: 'Answer engine',
              status: controller.hasLocalModel
                  ? 'Installed'
                  : answerProgress == null
                  ? 'Required'
                  : 'Downloading $answerProgress%',
              progress: answerProgress == null ? null : answerProgress / 100,
              complete: controller.hasLocalModel,
            ),
            const SizedBox(height: 12),
            _DependencyRow(
              icon: Icons.manage_search,
              title: 'Semantic search',
              status: !controller.supportsEmbeddingModel
                  ? 'Unavailable on this device'
                  : controller.hasEmbeddingModel
                  ? 'Installed'
                  : semanticProgress == null
                  ? 'Required'
                  : 'Downloading ${(semanticProgress * 100).round()}%',
              progress: semanticProgress,
              complete:
                  controller.supportsEmbeddingModel &&
                  controller.hasEmbeddingModel,
              disabled: !controller.supportsEmbeddingModel,
            ),
            if (controller.error != null) ...[
              const SizedBox(height: 16),
              Text(
                controller.error!,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ],
            const SizedBox(height: 22),
            FilledButton.icon(
              onPressed: busy || ready ? null : controller.installDependencies,
              icon: busy
                  ? const SizedBox.square(
                      dimension: 18,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.download),
              label: Text(
                ready
                    ? 'VaultIQ installed'
                    : busy
                    ? 'Installing VaultIQ'
                    : 'Install VaultIQ',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DependencyRow extends StatelessWidget {
  const _DependencyRow({
    required this.icon,
    required this.title,
    required this.status,
    required this.complete,
    this.progress,
    this.disabled = false,
  });

  final IconData icon;
  final String title;
  final String status;
  final bool complete;
  final double? progress;
  final bool disabled;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = disabled
        ? theme.disabledColor
        : complete
        ? theme.colorScheme.primary
        : theme.colorScheme.onSurfaceVariant;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: complete ? const Color(0xFFEAFBF5) : Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: complete ? const Color(0xFF8DE4D3) : _BrandColors.border,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          children: [
            Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: complete
                        ? _BrandColors.deepTeal
                        : _BrandColors.panelSoft,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                    icon,
                    color: complete ? Colors.white : color,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(child: Text(title, style: theme.textTheme.titleSmall)),
                Icon(
                  complete ? Icons.check_circle : Icons.radio_button_unchecked,
                  color: color,
                  size: 20,
                ),
              ],
            ),
            const SizedBox(height: 6),
            Row(
              children: [
                const SizedBox(width: 36),
                Expanded(child: Text(status, style: theme.textTheme.bodySmall)),
              ],
            ),
            if (progress != null) ...[
              const SizedBox(height: 10),
              LinearProgressIndicator(value: progress),
            ],
          ],
        ),
      ),
    );
  }
}

class _StatusHeader extends StatelessWidget {
  const _StatusHeader({required this.controller});

  final AppController controller;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xF20B2D32), Color(0xF20D6B62), Color(0xF25A4220)],
        ),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0x33FFFFFF)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x33000000),
            blurRadius: 18,
            offset: Offset(0, 10),
          ),
        ],
      ),
      child: Wrap(
        spacing: 12,
        runSpacing: 12,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: const Color(0x22FFFFFF),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0x33FFFFFF)),
            ),
            child: Icon(
              controller.error == null
                  ? Icons.shield_outlined
                  : Icons.error_outline,
              color: controller.error == null
                  ? _BrandColors.mint
                  : _BrandColors.coral,
            ),
          ),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 360),
            child: Text(
              controller.error ?? controller.status,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          _Metric(label: 'Docs', value: '${controller.stats.documentCount}'),
          _Metric(label: 'Chunks', value: '${controller.stats.chunkCount}'),
          _Metric(
            label: 'Vectors',
            value: '${controller.stats.embeddedChunkCount}',
          ),
        ],
      ),
    );
  }
}

class _ImportToolbar extends StatelessWidget {
  const _ImportToolbar({required this.controller});

  final AppController controller;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 4, 16, 8),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xEFFFFFFF),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0x66FFFFFF)),
      ),
      child: Row(
        children: [
          Expanded(
            child: FilledButton.icon(
              onPressed: controller.isIndexing ? null : controller.importFiles,
              icon: const Icon(Icons.note_add_outlined),
              label: const Text('Import files'),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: OutlinedButton.icon(
              onPressed: controller.isIndexing ? null : controller.importFolder,
              icon: const Icon(Icons.create_new_folder_outlined),
              label: const Text('Folder'),
            ),
          ),
          if (controller.isIndexing) ...[
            const SizedBox(width: 12),
            const SizedBox.square(
              dimension: 22,
              child: CircularProgressIndicator(strokeWidth: 2.4),
            ),
          ],
        ],
      ),
    );
  }
}

class _DocumentStrip extends StatelessWidget {
  const _DocumentStrip({required this.controller});

  final AppController controller;

  @override
  Widget build(BuildContext context) {
    if (controller.documents.isEmpty) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(16, 24, 16, 12),
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: _BrandColors.panel,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: const Color(0x66FFFFFF)),
          ),
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Row(
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: _BrandColors.panelSoft,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                    Icons.folder_open,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                ),
                const SizedBox(width: 12),
                const Expanded(
                  child: Text(
                    'No indexed files yet. Import files or a folder to start.',
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return SizedBox(
      height: 88,
      child: ListView.separated(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
        scrollDirection: Axis.horizontal,
        itemBuilder: (context, index) {
          final doc = controller.documents[index];
          return SizedBox(
            width: 230,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: _BrandColors.panel,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0x66FFFFFF)),
              ),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    Container(
                      width: 38,
                      height: 38,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [_BrandColors.mint, _BrandColors.sky],
                        ),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(
                        Icons.description_outlined,
                        color: _BrandColors.deepTeal,
                        size: 21,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            doc.name,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: Theme.of(context).textTheme.labelLarge,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '${doc.chunkCount} chunks',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
        separatorBuilder: (_, _) => const SizedBox(width: 10),
        itemCount: controller.documents.length,
      ),
    );
  }
}

class _ChatBubble extends StatelessWidget {
  const _ChatBubble({required this.turn});

  final ChatTurn turn;

  @override
  Widget build(BuildContext context) {
    final isUser = turn.role == ChatRole.user;
    final theme = Theme.of(context);
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 720),
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: isUser ? null : _BrandColors.panel,
            gradient: isUser
                ? const LinearGradient(
                    colors: [_BrandColors.teal, _BrandColors.deepTeal],
                  )
                : null,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: isUser ? const Color(0x6600FFE0) : const Color(0x66FFFFFF),
            ),
            boxShadow: const [
              BoxShadow(
                color: Color(0x22000000),
                blurRadius: 16,
                offset: Offset(0, 8),
              ),
            ],
          ),
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (turn.pending)
                  const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SizedBox.square(
                        dimension: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                      SizedBox(width: 10),
                      Text('Thinking locally'),
                    ],
                  )
                else
                  Text(
                    turn.text,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: isUser
                          ? theme.colorScheme.onPrimary
                          : _BrandColors.deepTeal,
                    ),
                  ),
                if (turn.sources.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: turn.sources
                        .take(4)
                        .map((hit) {
                          return Chip(
                            visualDensity: VisualDensity.compact,
                            avatar: const Icon(
                              Icons.article_outlined,
                              size: 16,
                            ),
                            label: Text(
                              hit.document.name,
                              overflow: TextOverflow.ellipsis,
                            ),
                          );
                        })
                        .toList(growable: false),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _QuestionBar extends StatelessWidget {
  const _QuestionBar({
    required this.controller,
    required this.enabled,
    required this.hintText,
    required this.onSubmit,
  });

  final TextEditingController controller;
  final bool enabled;
  final String hintText;
  final VoidCallback onSubmit;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 16),
      decoration: const BoxDecoration(
        color: Color(0xF4FFFFFF),
        border: Border(top: BorderSide(color: Color(0x66FFFFFF))),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: TextField(
              controller: controller,
              enabled: enabled,
              minLines: 1,
              maxLines: 4,
              textInputAction: TextInputAction.send,
              onSubmitted: enabled ? (_) => onSubmit() : null,
              decoration: InputDecoration(
                hintText: hintText,
                prefixIcon: const Icon(Icons.auto_awesome),
              ),
            ),
          ),
          const SizedBox(width: 10),
          IconButton.filled(
            tooltip: 'Ask',
            onPressed: enabled ? onSubmit : null,
            icon: const Icon(Icons.send),
          ),
        ],
      ),
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: 76,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0x20FFFFFF),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0x24FFFFFF)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            value,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: theme.textTheme.labelLarge?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w800,
            ),
          ),
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: theme.textTheme.bodySmall?.copyWith(
              color: const Color(0xCCFFFFFF),
            ),
          ),
        ],
      ),
    );
  }
}
